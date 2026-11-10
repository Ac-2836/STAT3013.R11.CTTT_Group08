"""Reproducible C3 analyses and battery-held-out regression forecasting."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from regression_sem_utils import (ROOT, DATA, TABLES, FIGURES, TRAIN, TEST, SCHEMA,
    load_cycles, regression_data, origin_position, metric_table, eol_record,
    save_table, validate_predictions, write_json)

MAIN = ['C(ambient_C)', 'current_c', 'cutoff_c']
INTERACTIONS = ['cycle_c:C(ambient_C)', 'cycle_c:current_c', 'cycle_c:cutoff_c']


def design_info(fit):
    # statsmodels 0.15 renamed data.design_info to data.model_spec.
    return getattr(fit.model.data, 'design_info', None) or fit.model.data.model_spec


def center(data, means=None):
    means = means or {c: float(data[c].mean()) for c in ['cycle_valid', 'I_level_A', 'discharge_cutoff_V']}
    x = data.copy()
    for original, new in [('cycle_valid', 'cycle_c'), ('I_level_A', 'current_c'), ('discharge_cutoff_V', 'cutoff_c')]:
        x[new] = x[original] - means[original]
    # Fix only the two design levels known from the locked Tier A protocol, not from test outcomes.
    x['ambient_C'] = pd.Categorical(x.ambient_C, categories=[4., 24.])
    return x, means


def formula(terms, response='ln_soh'):
    return response + ' ~ ' + ' + '.join(['cycle_c'] + terms)


def backward(data, covariance='cluster'):
    """Fit full model, test eligible terms, remove largest p>.05, refit/recheck.

    Interactions are removable separately. A main effect cannot be removed while
    any retained interaction contains it. Selection is exploratory, not confirmatory.
    """
    terms = MAIN + INTERACTIONS
    trace = []
    while True:
        fit = smf.ols(formula(terms), data=data).fit()
        infer = fit.get_robustcov_results(cov_type='cluster', groups=data.battery_id,
                use_correction=True, df_correction=True, use_t=True) if covariance == 'cluster' else fit
        eligible = []
        for term in terms:
            if ':' not in term and any(term in i.split(':') for i in terms if ':' in i):
                continue
            idx = design_info(fit).term_name_slices[term]
            restriction = np.eye(len(fit.params))[idx, :]
            p = float(np.asarray(infer.f_test(restriction).pvalue).squeeze())
            eligible.append((term, p))
        removable = [(t, p) for t, p in eligible if p > .05]
        removed, pmax = max(removable, key=lambda a: (a[1], a[0])) if removable else ('none', np.nan)
        trace.append(dict(step=len(trace), formula=formula(terms), covariance=covariance,
                          adjusted_R2=fit.rsquared_adj, AIC=fit.aic, BIC=fit.bic,
                          eligible_p_values=json.dumps(dict(eligible)), removed=removed, removed_p=pmax))
        if not removable:
            return fit, pd.DataFrame(trace)
        terms.remove(removed)


def coefficient_table(fit, groups=None, hc3=False):
    if groups is not None:
        robust = fit.get_robustcov_results(cov_type='cluster', groups=groups, use_t=True,
                                          use_correction=True, df_correction=True)
        label = 'cluster'
    elif hc3:
        robust = fit.get_robustcov_results(cov_type='HC3', use_t=True)
        label = 'HC3'
    else:
        robust, label = fit, 'OLS'
    ci = np.asarray(robust.conf_int())
    return pd.DataFrame(dict(term=fit.params.index, coefficient=fit.params.to_numpy(),
        OLS_SE=fit.bse.to_numpy(), OLS_p=fit.pvalues.to_numpy(), OLS_CI_low=fit.conf_int()[0],
        OLS_CI_high=fit.conf_int()[1], robust_type=label, robust_SE=robust.bse,
        robust_t=robust.tvalues, robust_p=robust.pvalues, robust_CI_low=ci[:, 0], robust_CI_high=ci[:, 1])).reset_index(drop=True)


def vif(fit):
    x = fit.model.exog
    return pd.DataFrame(dict(term=fit.model.exog_names, VIF=[variance_inflation_factor(x, i) for i in range(x.shape[1])]))


def diagnostic_plot(fit, data, name):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].scatter(fit.fittedvalues, fit.resid, s=8, alpha=.3)
    ax[0].axhline(0, color='k', lw=.7)
    ax[0].set(xlabel='Fitted response', ylabel='Residual')
    sm.qqplot(fit.resid, fit=True, line='45', ax=ax[1])
    ax[1].set_title('Standardized residual Q-Q')
    for bid, g in data.assign(residual=np.asarray(fit.resid)).groupby('battery_id'):
        ax[2].plot(g.cycle_valid, g.residual, lw=.7, label=bid)
    ax[2].set(xlabel='Valid cycle', ylabel='Residual')
    fig.tight_layout()
    fig.savefig(FIGURES / (name + '.png'), dpi=220)
    plt.show()
    plt.close(fig)
    bp = het_breuschpagan(fit.resid, fit.model.exog)
    jb = jarque_bera(fit.resid)
    dw = data.assign(residual=np.asarray(fit.resid)).groupby('battery_id').residual.apply(durbin_watson)
    save_table(dw.rename('Durbin_Watson').reset_index(), name + '_dw')
    return dict(BP_LM=bp[0], BP_p=bp[1], JB=jb[0], JB_p=jb[1],
                max_leverage=float(fit.get_influence().hat_matrix_diag.max()))


def chapter3(data):
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = {}
    train = data[data.battery_id.isin(TRAIN)]
    corr_cols = ['soh', 'cycle_valid', 'Re_ohm', 'Rct_ohm', 'T_max_C', 'chg_cc_time_s']
    for name, frame in [('train', train), ('g1_train', train[train.group.eq('G1')])]:
        out['correlation_' + name] = save_table(frame[corr_cols].corr().rename_axis('variable').reset_index(), 'regression_corr_' + name)
    # C3.2 is G1, as required. Test battery is excluded from all inferential fitting here.
    g1 = train[train.group.eq('G1')].copy()
    simple = smf.ols('soh ~ cycle_valid', data=g1).fit()
    out['simple_coefficients'] = save_table(coefficient_table(simple, g1.battery_id), 'regression_simple_g1_coefficients')
    out['simple_fit'] = save_table(pd.DataFrame([dict(n=int(simple.nobs), batteries=g1.battery_id.nunique(),
        R2=simple.rsquared, adjusted_R2=simple.rsquared_adj, F=simple.fvalue, F_p=simple.f_pvalue,
        **diagnostic_plot(simple, g1, 'regression_simple_diagnostics'))]), 'regression_simple_g1_fit')
    forms = [('linear', 'soh ~ cycle_valid'), ('loglinear', 'ln_soh ~ cycle_valid'),
             ('quadratic', 'soh ~ cycle_valid + I(cycle_valid ** 2)')]
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(g1.cycle_valid, g1.soh, s=7, alpha=.2)
    grid = pd.DataFrame({'cycle_valid': np.linspace(g1.cycle_valid.min(), g1.cycle_valid.max(), 200)})
    for name, f in forms:
        m = smf.ols(f, data=g1).fit()
        smear = float(np.mean(np.exp(m.resid))) if name == 'loglinear' else 1.
        pred = smear*np.exp(m.fittedvalues) if name == 'loglinear' else m.fittedvalues
        rows.append(dict(model=name, response='ln(SOH)' if name == 'loglinear' else 'SOH', n=len(g1),
            R2_response=m.rsquared, adjusted_R2_response=m.rsquared_adj, smearing=smear,
            RMSE_SOH=float(np.sqrt(np.mean((g1.soh-pred)**2))),
            AIC_SOHLikelihood=m.aic+2*np.log(g1.soh).sum() if name == 'loglinear' else m.aic,
            BIC_SOHLikelihood=m.bic+2*np.log(g1.soh).sum() if name == 'loglinear' else m.bic))
        gp = m.predict(grid)
        ax.plot(grid.cycle_valid, smear*np.exp(gp) if name == 'loglinear' else gp, label=name)
    ax.set(xlabel='Valid cycle', ylabel='SOH', title='G1 training batteries: functional forms')
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/'regression_functional_forms.png', dpi=220)
    plt.show(); plt.close(fig)
    out['functional_forms'] = save_table(pd.DataFrame(rows), 'regression_functional_forms')
    centered, means = center(train)
    full = smf.ols(formula(MAIN + INTERACTIONS), data=centered).fit()
    selected, trace = backward(centered)
    _, ols_trace = backward(centered, 'OLS')
    out['selection'] = save_table(trace, 'regression_backward_cluster')
    out['selection_OLS_sensitivity'] = save_table(ols_trace, 'regression_backward_ols')
    for name, fit in [('research_full', full), ('selected', selected)]:
        out[name + '_coefficients'] = save_table(coefficient_table(fit, train.battery_id), 'regression_' + name + '_coefficients')
        out[name + '_vif'] = save_table(vif(fit), 'regression_' + name + '_vif')
        out[name + '_fit'] = save_table(pd.DataFrame([dict(formula=fit.model.formula, R2=fit.rsquared,
            adjusted_R2=fit.rsquared_adj, F=fit.fvalue, F_p=fit.f_pvalue,
            **diagnostic_plot(fit, centered, 'regression_preferred_diagnostics' if name == 'research_full' else 'regression_selected_diagnostics'))]), 'regression_' + name + '_fit')
    # Supplementary Re/Rct VIF on complete training rows; no EIS imputation or target-based selection.
    eis = centered.dropna(subset=['Re_ohm', 'Rct_ohm'])
    eis_fit = smf.ols(formula(MAIN+INTERACTIONS) + ' + Re_ohm + Rct_ohm', data=eis).fit()
    out['eis_vif'] = save_table(vif(eis_fit), 'regression_eis_design_vif')
    effects = []
    for ambient in [4., 24.]:
        for current in sorted(train.I_level_A.unique()):
            # Report only observed temperature/current combinations.
            if not ((train.ambient_C == ambient) & (train.I_level_A == current)).any():
                continue
            q = pd.DataFrame({'cycle_valid': [means['cycle_valid'], means['cycle_valid']+1],
                              'I_level_A': current, 'discharge_cutoff_V': means['discharge_cutoff_V'], 'ambient_C': ambient})
            cq, _ = center(q, means)
            from patsy import build_design_matrices
            design = np.asarray(build_design_matrices([design_info(full)], cq)[0])
            robust = full.get_robustcov_results(cov_type='cluster', groups=train.battery_id,
                                               use_t=True, use_correction=True, df_correction=True)
            test = robust.t_test(design[1]-design[0])
            b = float(np.asarray(test.effect).item()); lo, hi = test.conf_int()[0]
            effects.append(dict(ambient_C=ambient, I_level_A=current, cutoff_V=means['discharge_cutoff_V'],
                log_slope=b, relative_pct_per_cycle=100*np.expm1(b), CI_low_pct=100*np.expm1(lo), CI_high_pct=100*np.expm1(hi),
                p=float(np.asarray(test.pvalue).item())))
    out['cycle_slopes'] = save_table(pd.DataFrame(effects), 'regression_conditional_cycle_slopes')
    fade = pd.read_csv(DATA/'battery_fade.csv')
    assert len(fade) == 23 and fade.battery_id.is_unique
    # Verify the supplied two-stage outcome without rewriting it.
    clean = load_cycles()
    frows = clean[clean.tier.isin(['A','B','N']) & ~clean.cap_outlier & ~clean.break_in]
    for _, r in fade.iterrows():
        g = frows[frows.battery_id.eq(r.battery_id)]
        np.testing.assert_allclose(-10000*np.polyfit(g.cycle_valid, g.soh, 1)[0], r.fade_pct_per_100cyc)
        w = g[g.cycle_valid <= 25]
        np.testing.assert_allclose(-10000*np.polyfit(w.cycle_valid, w.soh, 1)[0], r.fade25_pct_per_100cyc)
    for target in ['fade_pct_per_100cyc', 'fade25_pct_per_100cyc']:
        m = smf.ols(target+' ~ ambient_C + I_level_A + discharge_cutoff_V', data=fade).fit()
        out[target] = save_table(coefficient_table(m, hc3=True), 'regression_' + target)
    out['fade_sample'] = save_table(fade, 'regression_battery_fade_verified')
    return out


def fit_forecasters(train):
    x, means = center(train)
    selected, trace = backward(x)
    full = smf.ols(formula(MAIN+INTERACTIONS), data=x).fit()
    quadratic = smf.ols(formula(MAIN+INTERACTIONS+['I(cycle_c ** 2)']), data=x).fit()
    return dict(means=means, fits={'research_loglinear': full, 'selected_loglinear': selected,
                                 'quadratic_loglinear': quadratic}, selection=trace)


VARIANTS = ['research_loglinear', 'selected_loglinear', 'quadratic_loglinear',
            'anchored_loglinear', 'anchored_quadratic']
OPERATING_CONDITIONS = ['ambient_C', 'I_level_A', 'discharge_cutoff_V']
HISTORICAL_ARCHIVE = ROOT/'reports/audit/regression_sem_before/regression.csv'


def operating_condition_check(data):
    """Fail closed unless the fixed Tier A design is verified for every row.

    This is an offline availability check, not an estimate of a future schedule.
    Time-varying conditions require a separately agreed known-at-origin protocol.
    """
    rows = []
    for bid, g in data.groupby('battery_id', sort=True):
        for feature in OPERATING_CONDITIONS:
            values = g[feature].to_numpy(dtype=float)
            fixed = bool(np.isfinite(values).all() and (values == values[0]).all())
            rows.append(dict(battery_id=bid, feature=feature, n=len(g),
                             unique_values=g[feature].nunique(dropna=False), fixed=fixed))
    checks = pd.DataFrame(rows)
    if checks.empty or not checks.fixed.all():
        bad = checks.loc[~checks.fixed, ['battery_id', 'feature']].to_dict('records') if not checks.empty else []
        raise ValueError(f'E2 feature availability requires fixed operating conditions; '
                         f'unknown future conditions are unsupported: {bad}')
    return checks


def forecast(bundle, evaluation):
    parts = []
    for bid, g in evaluation.groupby('battery_id', sort=True):
        g = g.sort_values('cycle_valid').reset_index(drop=True)
        k = origin_position(len(g)); tail = g.iloc[k:].copy()
        operating_condition_check(g)
        for scenario in ['E1', 'E2']:
            predictors = g.copy()
            if scenario == 'E2':
                # Read condition values at the origin only, even after validation.
                for feature in OPERATING_CONDITIONS:
                    predictors[feature] = g[feature].iloc[k-1]
            cx, _ = center(predictors, bundle['means'])
            for variant in VARIANTS + ['persistence']:
                if variant == 'persistence':
                    yp = g.soh.shift().iloc[k:].to_numpy() if scenario == 'E1' else np.full(len(tail), g.soh.iloc[k-1])
                else:
                    base = {'anchored_loglinear': 'research_loglinear', 'anchored_quadratic': 'quadratic_loglinear'}.get(variant, variant)
                    fit = bundle['fits'][base]
                    mu = np.asarray(fit.predict(cx))
                    if variant.startswith('anchored_'):
                        # Intercept recalibration on permitted history; global Duan factors cancel in ratios.
                        reference = np.arange(k-1, len(g)-1) if scenario == 'E1' else np.full(len(tail), k-1)
                        yp = g.soh.to_numpy()[reference]*np.exp(mu[k:]-mu[reference])
                    else:
                        yp = np.exp(mu[k:])*np.mean(np.exp(fit.resid))
                p = pd.DataFrame(dict(model=variant, battery_id=bid, cycle_valid=tail.cycle_valid.to_numpy(),
                        scenario=scenario, y_true=tail.soh.to_numpy(), y_pred=yp, y_lower=np.nan, y_upper=np.nan))
                parts.append(p)
    return pd.concat(parts, ignore_index=True)


def historical_predictions(data, archive_path):
    """Optional validated comparison; never an input to fitting or selection."""
    try:
        before = pd.read_csv(archive_path)
        validate_predictions(before, data)
        if before.model.nunique() != 1:
            raise ValueError('Historical archive must contain one model')
        before['model'] = 'preaudit_working_copy'
        return before, 'available'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, AssertionError) as exc:
        status = f'skipped: {type(exc).__name__}: {exc}'
        print('Historical regression comparison '+status, flush=True)
        return None, status


def evaluate_regression(data, archive_path=HISTORICAL_ARCHIVE):
    # Audit the entire Tier A series, including rows excluded from estimation.
    cycles = load_cycles()
    save_table(operating_condition_check(cycles[cycles.tier.eq('A')]),
               'regression_operating_conditions')
    train, test = data[data.battery_id.isin(TRAIN)].copy(), data[data.battery_id.isin(TEST)].copy()
    cv_rows, predictions, selection_rows, split_rows = [], [], [], []
    for fold, (it, iv) in enumerate(GroupKFold(4).split(train, groups=train.battery_id)):
        fit_data, validation = train.iloc[it], train.iloc[iv]
        assert set(fit_data.battery_id).isdisjoint(validation.battery_id)
        bundle = fit_forecasters(fit_data)
        p = forecast(bundle, validation)
        p['fold'] = fold; predictions.append(p)
        m = metric_table(p); m['fold'] = fold; cv_rows.append(m)
        trace = bundle['selection'].copy(); trace['fold'] = fold; selection_rows.append(trace)
        split_rows.append(dict(fold=fold, training_batteries=','.join(sorted(fit_data.battery_id.unique())),
                               validation_batteries=','.join(sorted(validation.battery_id.unique())), means=json.dumps(bundle['means'])))
    cv = pd.concat(cv_rows, ignore_index=True)
    save_table(cv, 'regression_cv_metrics')
    save_table(pd.concat(predictions), 'regression_cv_predictions')
    save_table(pd.concat(selection_rows), 'regression_cv_backward')
    save_table(pd.DataFrame(split_rows), 'regression_cv_splits')
    # Freeze winners using equal weight per held-out battery, before opening test outcomes.
    scores = cv[(cv.aggregation == 'battery') & cv.model.isin(VARIANTS)].groupby(['scenario','model'], as_index=False).MAE.mean()
    winners = {s: g.sort_values(['MAE','model']).iloc[0].model for s,g in scores.groupby('scenario')}
    save_table(scores, 'regression_cv_selection')
    bundle = fit_forecasters(train)
    save_table(bundle['selection'], 'regression_train_backward')
    write_json(TABLES/'regression_selected.json', dict(winners=winners, means=bundle['means'],
        formulas={k: v.model.formula for k,v in bundle['fits'].items()},
        strategy='scenario-specific regression forecasting; separate fitted variants',
        exported_model='regression',
        anchors={'E1': 'preceding retained observed SOH of this evaluation battery',
                 'E2': 'fixed origin observed SOH of this evaluation battery'},
        condition_policy='fixed Tier A conditions validated; E2 values read at origin only',
        selection_metric='mean held-out battery MAE; GroupKFold(4), training batteries only'))
    all_preds = forecast(bundle, test)
    validate_predictions(all_preds[SCHEMA], data)
    save_table(all_preds, 'regression_test_variants_predictions')
    chosen = pd.concat([all_preds[(all_preds.scenario == s) & (all_preds.model == name)] for s,name in winners.items()]).sort_values(['battery_id','scenario','cycle_valid'])
    chosen['model'] = 'regression'
    comparison = pd.concat([all_preds, chosen], ignore_index=True)
    results = save_table(metric_table(comparison), 'regression_test_metrics')
    # Forecast origins, explicit censoring, and no extrapolation.
    eols = []
    for (model, bid), p in comparison[comparison.scenario.eq('E2')].groupby(['model','battery_id']):
        g = test[test.battery_id.eq(bid)].sort_values('cycle_valid')
        history = g.iloc[:origin_position(len(g))]
        eols.append(dict(model=model, battery_id=bid, **eol_record(history, p.sort_values('cycle_valid'))))
    eol = save_table(pd.DataFrame(eols), 'regression_eol_rul')
    validate_predictions(chosen[SCHEMA], data)
    chosen[SCHEMA].to_csv(ROOT/'reports'/'predictions'/'regression.csv', index=False)
    # Save the research baseline outside predictions/ to avoid duplicate rows in group comparison.
    save_table(all_preds[all_preds.model.eq('research_loglinear')], 'regression_research_predictions')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, (bid, g) in zip(axes.flat, test.groupby('battery_id', sort=True)):
        k = origin_position(len(g)); ax.plot(g.cycle_valid, g.soh, color='k', lw=1.2, label='Observed')
        for s, style in [('E1','-'), ('E2','--')]:
            p = chosen[(chosen.battery_id == bid) & (chosen.scenario == s)]
            ax.plot(p.cycle_valid, p.y_pred, style, lw=1.2, label=s+': '+winners[s])
        research = all_preds[(all_preds.battery_id == bid) & (all_preds.scenario == 'E2') & all_preds.model.eq('research_loglinear')]
        ax.plot(research.cycle_valid, research.y_pred, ':', color='gray', label='Research log-linear')
        ax.axhline(.8, color='red', lw=.8, label='EOL = 0.8')
        ax.axvline(g.cycle_valid.iloc[k-1], color='gray', lw=.7)
        ax.set(title=bid, xlabel='Valid cycle', ylabel='SOH'); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIGURES/'regression_test_forecasts.png', dpi=260)
    plt.show(); plt.close(fig)
    # Quantify extrapolation rather than attribute failure to it without evidence.
    range_rows = []
    for bid, g in test.groupby('battery_id'):
        for col in ['cycle_valid','ambient_C','I_level_A','discharge_cutoff_V']:
            range_rows.append(dict(battery_id=bid, feature=col, train_min=train[col].min(), train_max=train[col].max(),
                test_min=g[col].min(), test_max=g[col].max(), outside_global_train=int((~g[col].between(train[col].min(),train[col].max())).sum())))
    save_table(pd.DataFrame(range_rows), 'regression_extrapolation')
    # Comparison happens only after all current results have been generated.
    before, status = historical_predictions(data, archive_path)
    save_table(metric_table(before) if before is not None else results.iloc[:0],
               'regression_historical_metrics')
    write_json(TABLES/'regression_historical_status.json', dict(status=status))
    return dict(winners=winners, cv=scores, test_metrics=results, eol=eol)
