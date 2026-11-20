"""Observed-variable SEM, explicit exogenous moments, and battery-block uncertainty."""
from collections import Counter
from itertools import combinations
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from semopy import Model, calc_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf

from regression_sem_utils import ROOT, DATA, TABLES, FIGURES, load_cycles, save_table, write_json

CONDITIONS = ['ambient_C', 'I_level_A', 'discharge_cutoff_V']
EXOGENOUS = CONDITIONS + ['cycle_valid']
VARIABLES = EXOGENOUS + ['Re_ohm', 'Rct_ohm', 'ln_soh']
ORIGINAL = '''
Re_ohm ~ ambient_C + I_level_A + discharge_cutoff_V
Rct_ohm ~ ambient_C + I_level_A + discharge_cutoff_V
ln_soh ~ Re_ohm + Rct_ohm + ambient_C + I_level_A + discharge_cutoff_V + cycle_valid
'''
# Free the ten nuisance exogenous moments. They already match the empirical
# covariance by default in semopy, but are not counted in its parameter-based df.
EXOG_MOMENTS = '\n'.join([f'{a} ~~ {a}' for a in EXOGENOUS] +
                          [f'{a} ~~ {b}' for a,b in combinations(EXOGENOUS, 2)])
PRIMARY = ORIGINAL + EXOG_MOMENTS
AGE_CONTROL = PRIMARY.replace('Re_ohm ~ ambient_C', 'Re_ohm ~ cycle_valid + ambient_C').replace(
                              'Rct_ohm ~ ambient_C', 'Rct_ohm ~ cycle_valid + ambient_C')
SATURATED = AGE_CONTROL + '\nRe_ohm ~~ Rct_ohm'


def prepare_sem(cycles, export=True):
    save = save_table if export else lambda frame, name: frame
    source = cycles[cycles.tier.isin(['A','B'])].sort_values(['battery_id','cycle_valid']).copy()
    linked = source.dropna(subset=['eis_op_index']).drop_duplicates(['battery_id','eis_op_index']).copy()
    raw = pd.read_csv(DATA/'impedance_ops.csv')
    raw = raw[['battery_id','op_index','Re_ohm','Rct_ohm','ambient_C','start_time']].rename(columns={
        'op_index':'eis_op_index', 'Re_ohm':'raw_Re', 'Rct_ohm':'raw_Rct',
        'ambient_C':'eis_ambient_C', 'start_time':'eis_start_time'})
    linked = linked.merge(raw, on=['battery_id','eis_op_index'], validate='one_to_one', how='left')
    # Use the actual operation's measurements: a new index with carried-forward
    # cleaned resistance must never masquerade as a fresh EIS measurement.
    valid = linked.raw_Re.between(0,.5,inclusive='neither') & linked.raw_Rct.between(0,.5,inclusive='neither')
    missing = linked[VARIABLES[:-1]+['soh']].isna().sum()
    save(missing.rename('missing_before_complete_case').rename_axis('variable').reset_index(), 'sem_missingness')
    x = linked[valid].dropna(subset=EXOGENOUS+['soh']).copy()
    assert (x.soh > 0).all()
    assert not x.duplicated(['battery_id','eis_op_index']).any()
    assert (pd.to_datetime(x.eis_start_time) < pd.to_datetime(x.start_time)).all()
    mismatched = (~np.isclose(x.raw_Re,x.Re_ohm) | ~np.isclose(x.raw_Rct,x.Rct_ohm)).sum()
    x['Re_ohm'], x['Rct_ohm'] = x.raw_Re, x.raw_Rct
    x['ln_soh'] = np.log(x.soh)
    flow = pd.DataFrame([dict(stage='Tier A/B cycle rows', n=len(source)),
        dict(stage='Distinct linked EIS operations', n=len(linked)),
        dict(stage='Raw EIS invalid or missing', n=int((~valid).sum())),
        dict(stage='Analysis complete cases', n=len(x)), dict(stage='Independent battery blocks', n=x.battery_id.nunique()),
        dict(stage='Cleaned/raw resistance mismatches replaced', n=int(mismatched)),
        dict(stage='Flagged rows retained in historical/theory sample', n=int((x.cap_outlier|x.break_in).sum()))])
    save(flow, 'sem_sample_flow')
    save(x[['battery_id','cycle_valid','eis_op_index','ambient_C','eis_ambient_C','I_level_A',
                  'discharge_cutoff_V','Re_ohm','Rct_ohm','soh','ln_soh','cap_outlier','break_in']], 'sem_analysis_data')
    save(x.groupby(['ambient_C','eis_ambient_C']).size().rename('n').reset_index(), 'sem_eis_temperature')
    save(x[VARIABLES].corr().rename_axis('variable').reset_index(), 'sem_correlations')
    return x, flow


def fit_sem(data, description=PRIMARY, standardize=True, inspect=True):
    values = data[VARIABLES].astype(float)
    sd = values.std(ddof=0) if standardize else pd.Series(1., index=VARIABLES)
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError('Zero/invalid variance in bootstrap sample')
    z = (values-values.mean())/sd if standardize else values
    if np.linalg.matrix_rank(z.cov().to_numpy()) < len(VARIABLES):
        raise ValueError('Rank-deficient sample covariance')
    model = Model(description)
    options = {'maxiter':2000, 'ftol':1e-10} if standardize else {'maxiter':2000}
    result = model.fit(z, obj='MLW', solver='SLSQP', options=options)
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(model.param_vals).all():
        raise RuntimeError('Nonconverged/nonfinite optimizer: '+str(result.message))
    sigma = model.calc_sigma()[0]
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError('Non-positive-definite fitted covariance')
    if not inspect:
        paths = model.inspect(std_est=True, information=None)
    else:
        paths = model.inspect(std_est=True)
    paths['Estimate_optimizer_scale'] = paths.Estimate
    for i, r in paths.iterrows():
        factor = sd[r.lval]/sd[r.rval] if r.op == '~' else sd[r.lval]*sd[r.rval]
        paths.loc[i, 'Estimate'] = r.Estimate*factor
        if inspect:
            for c in ['Std. Err']:
                v = pd.to_numeric(pd.Series([r[c]]), errors='coerce').iloc[0]
                paths.loc[i,c] = v*abs(factor)
    if inspect:
        paths['CI_low_iid'] = paths.Estimate-1.959963984540054*pd.to_numeric(paths['Std. Err'], errors='coerce')
        paths['CI_high_iid'] = paths.Estimate+1.959963984540054*pd.to_numeric(paths['Std. Err'], errors='coerce')
    return model, result, paths


def effects(paths):
    def b(y,x):
        r = paths[(paths.lval==y) & paths.op.eq('~') & (paths.rval==x)]
        assert len(r)==1
        return float(r.Estimate.iloc[0])
    out = []
    for x in CONDITIONS:
        direct = b('ln_soh',x)
        via_re = b('Re_ohm',x)*b('ln_soh','Re_ohm')
        via_rct = b('Rct_ohm',x)*b('ln_soh','Rct_ohm')
        out.append(dict(predictor=x, direct=direct, indirect_via_Re=via_re, indirect_via_Rct=via_rct,
                        indirect_combined=via_re+via_rct, total=direct+via_re+via_rct))
    return pd.DataFrame(out)


def model_diagnostics(model, result, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        indices = calc_stats(model).iloc[0].to_dict()
    # Observed information rank is a numerical local identification check.
    fim = model.calc_fim()
    count = len(model.param_vals)
    info = dict(model=name, converged=bool(result.success), iterations=int(result.n_it),
                objective=float(result.fun), free_parameters=count,
                information_rank=int(np.linalg.matrix_rank(fim)), observed_moments=28, **indices)
    # Fit of saturated models is algebraic, not empirical evidence.
    if info['DoF'] == 0:
        for c in ['RMSEA','CFI','TLI','chi2 p-value']:
            info[c] = np.nan
    return info


def block_bootstrap(data, target=200, max_attempts=500):
    rng = np.random.default_rng(42)
    ids = np.array(sorted(data.battery_id.unique()))
    blocks = {bid:g.copy() for bid,g in data.groupby('battery_id')}
    all_effects, all_paths, audit = [], [], []
    valid, attempts = 0, 0
    while valid < target and attempts < max_attempts:
        sampled = rng.choice(ids, size=len(ids), replace=True)
        # Repeated choices duplicate the complete battery block, preserving its multiplicity.
        sample = pd.concat([blocks[bid].assign(bootstrap_block=j) for j,bid in enumerate(sampled)], ignore_index=True)
        attempts += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                model, result, path = fit_sem(sample, inspect=False)
            regressions = path[path.op.eq('~')].copy()
            if not np.isfinite(regressions[['Estimate','Est. Std']].to_numpy()).all():
                raise ValueError('Nonfinite path estimate')
            # Boundary/Heywood cases do not count as valid replicates.
            variances = path[(path.op=='~~') & (path.lval==path.rval)]
            if (variances.Estimate <= 0).any():
                raise ValueError('Nonpositive variance')
            ef = effects(path); ef['replicate'] = valid; all_effects.append(ef)
            regressions['replicate'] = valid; all_paths.append(regressions)
            audit.append(dict(attempt=attempts, valid=True, unique_batteries=len(set(sampled)),
                              sampled_batteries=','.join(sampled), iterations=int(result.n_it), error=''))
            valid += 1
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            audit.append(dict(attempt=attempts, valid=False, unique_batteries=len(set(sampled)),
                              sampled_batteries=','.join(sampled), iterations=np.nan, error=type(exc).__name__+': '+str(exc)))
        if attempts % 50 == 0:
            print(f'Bootstrap: {valid} valid / {attempts} attempts', flush=True)
    save_table(pd.DataFrame(audit), 'sem_bootstrap_audit')
    if valid < target:
        raise RuntimeError(f'Only {valid} valid bootstrap replicates after {attempts} attempts')
    replicates = pd.concat(all_effects, ignore_index=True)
    paths = pd.concat(all_paths, ignore_index=True)
    save_table(replicates, 'sem_bootstrap_effect_replicates')
    save_table(paths[['replicate','lval','op','rval','Estimate','Est. Std']], 'sem_bootstrap_path_replicates')
    interval_rows = []
    for predictor, g in replicates.groupby('predictor'):
        for kind in ['direct','indirect_via_Re','indirect_via_Rct','indirect_combined','total']:
            lo,hi = np.quantile(g[kind],[.025,.975])
            interval_rows.append(dict(predictor=predictor, effect=kind, bootstrap_SE=g[kind].std(ddof=1),
                                      CI_low=lo, CI_high=hi, excludes_zero=bool(lo>0 or hi<0), valid_replicates=valid))
    path_rows = []
    for (lhs,rhs), g in paths.groupby(['lval','rval']):
        lo,hi = np.quantile(g.Estimate,[.025,.975])
        slo,shi = np.quantile(g['Est. Std'],[.025,.975])
        path_rows.append(dict(lval=lhs, rval=rhs, bootstrap_SE=g.Estimate.std(ddof=1),
                              bootstrap_CI_low=lo, bootstrap_CI_high=hi,
                              standardized_CI_low=slo, standardized_CI_high=shi))
    summary = dict(seed=42, target_valid=target, valid=valid, attempts=attempts, failures=attempts-valid,
                   failure_reasons=dict(Counter(r['error'] for r in audit if not r['valid'])))
    write_json(TABLES/'sem_bootstrap_summary.json', summary)
    return save_table(pd.DataFrame(interval_rows), 'sem_effect_intervals'), save_table(pd.DataFrame(path_rows), 'sem_path_intervals'), summary


def path_diagram(paths):
    """One arrow per fitted regression; labels are looked up from actual estimates."""
    positions = {'ambient_C':(.08,.84), 'I_level_A':(.08,.53), 'discharge_cutoff_V':(.08,.22),
                 'Re_ohm':(.49,.75), 'Rct_ohm':(.49,.31), 'ln_soh':(.91,.53), 'cycle_valid':(.88,.10)}
    labels = {'ambient_C':'Ambient temperature\n(degrees C)', 'I_level_A':'Discharge current\n(A)',
              'discharge_cutoff_V':'Cutoff voltage\n(V)', 'Re_ohm':'Re (ohm)', 'Rct_ohm':'Rct (ohm)',
              'ln_soh':'ln(SOH)', 'cycle_valid':'Valid cycle'}
    fig,ax = plt.subplots(figsize=(13,7.5)); ax.set(xlim=(-.03,1.04),ylim=(-.05,1.04)); ax.axis('off')
    fig.subplots_adjust(left=.025,right=.975,top=.88,bottom=.12)
    regressions = paths[paths.op.eq('~')]
    for _,r in regressions.iterrows():
        start,end = positions[r.rval],positions[r.lval]
        direct = r.lval=='ln_soh' and r.rval in CONDITIONS
        rad = {'ambient_C':-.31,'I_level_A':.40,'discharge_cutoff_V':.31}[r.rval] if direct else 0
        if r.rval=='cycle_valid': rad=-.1
        color = '#b04a22' if direct else '#275c83'
        arrow = FancyArrowPatch(start,end,connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
            mutation_scale=12, lw=1.1, color=color, alpha=.85, shrinkA=35, shrinkB=31)
        ax.add_patch(arrow)
        if direct or r.rval=='cycle_valid':
            # Read the actual Bezier in data coordinates after display-space
            # curvature and endpoint shrinking; aspect ratio matters here.
            a,control,b = arrow.get_path().vertices[:3]
            t = {'ambient_C':.65,'I_level_A':.60,'discharge_cutoff_V':.59}.get(r.rval,.45)
            where = (1-t)**2*a+2*t*(1-t)*control+t*t*b
        else:
            fraction = {'ambient_C':.30,'I_level_A':.49,'discharge_cutoff_V':.70}.get(r.rval,.48)
            where = tuple(np.array(start)*(1-fraction)+np.array(end)*fraction)
        ax.text(*where, f'{r["Est. Std"]:.3f}', fontsize=9, ha='center',va='center',color=color,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
    for v,(x,y) in positions.items():
        ax.text(x,y,labels[v],ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.5',fc='#eef5fa',ec='#275c83'))
    ax.set_title('Operating conditions, impedance and SOH: primary observed-variable model', fontsize=13,pad=20)
    fig.text(.5,.025,'Arrow labels: standardized coefficients. Orange: direct condition paths.\n'
        'Exogenous variances/covariances freely estimated; omitted from drawing. Associations are not causal effects.',
        ha='center',fontsize=9)
    fig.savefig(FIGURES/'sem_path_diagram.png',dpi=320)
    fig.savefig(FIGURES/'sem_path_diagram.pdf')
    plt.show(); plt.close(fig)
    save_table(regressions[['lval','rval','Estimate','Est. Std']], 'sem_diagram_coefficients')


def run_sem(data):
    rows = []; primary_model = primary_paths = None
    specs = [('original_scaled_implicit_exog', ORIGINAL, True, data),
             ('primary_explicit_exog', PRIMARY, True, data),
             ('age_control_sensitivity', AGE_CONTROL, True, data),
             ('age_correlated_residuals_SATURATED', SATURATED, True, data),
             ('exclude_quality_flags_sensitivity', PRIMARY, True, data[~data.cap_outlier & ~data.break_in])]
    for name, desc, scaled, frame in specs:
        model, result, paths = fit_sem(frame, desc, scaled)
        row = model_diagnostics(model, result, name)
        row.update(n=len(frame), batteries=frame.battery_id.nunique())
        rows.append(row)
        save_table(paths, 'sem_'+name+'_coefficients')
        if name=='primary_explicit_exog':
            primary_model, primary_paths = model, paths
        print(name, {k:row[k] for k in ['n','DoF','chi2','CFI','TLI','RMSEA','information_rank','free_parameters']},flush=True)
    fit = save_table(pd.DataFrame(rows), 'sem_fit_comparison')
    distributions = data[VARIABLES].agg(['min','max','mean','std','skew','kurt']).T.rename_axis('variable').reset_index()
    save_table(distributions, 'sem_distributions')
    vif_rows = []
    for outcome,predictors in [('Re_ohm',CONDITIONS), ('Rct_ohm',CONDITIONS),
                               ('ln_soh',['Re_ohm','Rct_ohm']+EXOGENOUS)]:
        design = smf.ols(outcome+' ~ '+' + '.join(predictors),data=data).exog
        for j,name in enumerate(['Intercept']+predictors):
            vif_rows.append(dict(equation=outcome,predictor=name,VIF=variance_inflation_factor(design,j)))
    save_table(pd.DataFrame(vif_rows), 'sem_equation_vif')
    # Saturated sensitivity diagnoses omitted covariance/age, without promoting a model by its fit.
    sigma = primary_model.calc_sigma()[0]
    obs = primary_model.mx_cov
    scale = np.sqrt(np.outer(np.diag(obs),np.diag(obs)))
    resid = (obs-sigma)/scale
    names = primary_model.vars['observed']
    save_table(pd.DataFrame(resid,index=names,columns=names).rename_axis('variable').reset_index(), 'sem_standardized_covariance_residuals')
    effect_table = save_table(effects(primary_paths), 'sem_effects')
    intervals, path_intervals, bootstrap = block_bootstrap(data)
    regressions = primary_paths[primary_paths.op.eq('~')].merge(path_intervals,on=['lval','rval'],validate='one_to_one')
    save_table(regressions, 'sem_primary_paths_with_bootstrap')
    path_diagram(primary_paths)
    return dict(fit=fit, coefficients=regressions, effects=effect_table, intervals=intervals, bootstrap=bootstrap)
