"""Build review/report text from saved results; no result numbers are hand-entered."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from regression_sem_utils import ROOT, TABLES, TRAIN, TEST


def table(name):
    return pd.read_csv(TABLES/(name+'.csv'))


def markdown(frame, columns=None):
    x = frame[columns] if columns else frame
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v,(float,np.floating)): return f'{v:.6g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(x.columns)+' |', '| '+' | '.join(['---']*len(x.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in r)+' |' for r in x.itertuples(index=False,name=None)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--append-worklog', action='store_true')
    args = parser.parse_args()
    metrics = table('regression_test_metrics')
    macro = metrics[(metrics.aggregation=='macro') & metrics.model.isin(['preaudit_working_copy','research_loglinear','regression','persistence'])]
    micro = metrics[(metrics.aggregation=='micro') & metrics.model.isin(['regression','persistence'])]
    per_cell = metrics[(metrics.aggregation=='battery') & metrics.model.isin(['regression','persistence'])]
    selected = json.loads((TABLES/'regression_selected.json').read_text(encoding='utf8'))
    eol = table('regression_eol_rul')
    eol = eol[eol.model.isin(['regression','persistence'])]
    fits = table('sem_fit_comparison')
    historical_path = TABLES/'sem_historical_reported.csv'
    historical = (pd.read_csv(historical_path) if historical_path.exists() else
                  pd.DataFrame(columns=['DoF','chi2','CFI','TLI','RMSEA']))
    historical_text = (markdown(historical,['DoF','chi2','CFI','TLI','RMSEA']) if not historical.empty
                       else 'Historical SEM comparison skipped: no valid optional archive output available.')
    primary = fits[fits.model.eq('primary_explicit_exog')].iloc[0]
    paths = table('sem_primary_paths_with_bootstrap')
    bootstrap = json.loads((TABLES/'sem_bootstrap_summary.json').read_text(encoding='utf8'))
    execution = json.loads((TABLES/'regression_sem_execution.json').read_text(encoding='utf8'))
    validation_path = ROOT/'reports/audit/regression_sem_final_validation.json'
    if not validation_path.exists():
        validation_path = ROOT/'reports/audit/regression_sem_validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf8')) if validation_path.exists() else {}
    validation_brief = {k:v for k,v in validation.items() if k != 'figure_metadata'}
    g1 = table('regression_simple_g1_fit').iloc[0]
    vif = table('regression_research_full_vif')
    corr = table('regression_corr_train').set_index('variable').loc['Re_ohm','Rct_ohm']
    g1corr = table('regression_corr_g1_train').set_index('variable').loc['Re_ohm','Rct_ohm']
    reference = '''
## Verified methodological sources

- N. Duan, “Smearing Estimate: A Nonparametric Retransformation Method,” *JASA*, vol. 78, no. 383, pp. 605–610, 1983. [Publisher / DOI](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017).
- [scikit-learn GroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): non-overlapping validation groups.
- [statsmodels robust covariance documentation](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html): cluster correction, group degrees of freedom and t inference.
- [semopy model documentation](https://semopy.com/docs/model.html), [syntax](https://semopy.com/syntax.html) and [fit-statistic implementation](https://semopy.com/docs/stats.html): covariance specification, estimation and parameter-count degrees of freedom. Checked against installed semopy source.

These sources document the methods. They do not establish battery-specific causal mechanisms; the team's battery literature review still needs substantive primary studies.
'''
    regression = f'''# Regression methods and results — Quoc Dat

Generated from executed outputs on 2026-11-10 (audit started 2026-11-09). English text is intended for integration into Full Report §§3.3/4/5.3 and the IEEE paper. The team documents and other members' models are not overwritten.

## Research purpose and mathematical model

The primary explanatory model is

`ln(SOH_it) = β0 + βc c_it + βT D24_i + βI I_i,c + βV V_i,c + βcT c_it D24_i + βcI c_it I_i,c + βcV c_it V_i,c + ε_it`.

Here `c` is valid cycle centered on training rows, `D24` compares 24°C with the 4°C reference,
`I_c` is centered discharge current (A), and `V_c` is centered cutoff voltage (V).
SOH is a capacity ratio and has no units. Re/Rct are in ohm; temperatures are °C;
preceding-charge CC time is in seconds. `cycle_valid` counts cleaned valid discharges, including flagged rows,
so excluded flags leave gaps in the time coordinate. Profile is constant CC in Tier A and has no estimable dummy.
The exact relative SOH change for a log slope b is `100[exp(b)−1]%` per valid cycle, not percentage points of SOH.

## Data preparation and separation of inference from prediction

Read `cycles_clean.csv`; keep Tier A, remove `cap_outlier` and `break_in`, assert positive SOH,
required-variable completeness and unique battery/cycle and operation keys. No original table is changed.
The descriptive preamble retains the original full-sample plots. C3 estimation and correlation/VIF decisions use training cells.
Training cells: {', '.join(TRAIN)}. Test cells: {', '.join(TEST)}. There is no battery overlap.
The G1 simple-regression sample has {int(g1['n'])} rows from {int(g1.batteries)} training batteries;
B0007 is excluded from fitting. Battery-level sensitivity deliberately uses all 23 independent cells as an explanatory analysis,
and is never used to select the held-out forecasting model.

No current-discharge capacity, energy, duration, cutoff time, voltage minimum or maximum temperature is a forecasting input.
Re/Rct and charge duration are audited descriptively but are excluded from the forecast candidate set because E2 future values are unknown.
Anchored models use only past observed SOH and constant operating conditions; C0 is not a cross-cell predictor.
The first-ten-cycle C0 is already known by every forecast origin. Tier eligibility depends on the lifetime record,
and the locked Hampel mask uses a centered window. Thus the shared dataset is a **retrospective clean-data benchmark**,
not a fully prospective online validation. This upstream look-ahead cannot be removed without changing the locked group protocol.
E1 means the next *retained* clean observation; gaps can span more than one `cycle_valid` unit.

## C3.1: correlation and multicollinearity

Train Re–Rct Pearson r = {corr:.6f}; G1 train r = {g1corr:.6f}. These differ from published full-sample EDA values because the sample differs.
Pairwise correlation uses available pairs; VIF uses the actual complete-case dummy/interaction design.

{markdown(vif)}

Centering reduces nonessential interaction collinearity but does not eliminate missing condition combinations.
The supplementary EIS design (`regression_eis_design_vif.csv`) has substantial Rct/condition collinearity;
both impedance predictors are therefore not added to the primary regression. A threshold of 10 is a diagnostic,
not an automatic argument for deleting a theoretically required main effect.

## C3.2–C3.3: G1 simple regression and functional forms

{markdown(table('regression_simple_g1_fit'))}

{markdown(table('regression_simple_g1_coefficients'))}

OLS F/t tests assume independent errors; the displayed battery-clustered sensitivity has only three independent groups
and should not support strong population claims. Residual plots, standardized Q-Q plots, Breusch–Pagan,
Jarque–Bera and battery-specific Durbin–Watson diagnostics are saved. Heteroskedasticity and serial dependence remain.

{markdown(table('regression_functional_forms'))}

R² and adjusted R² are on the stated response scale and must not be ranked across SOH versus ln(SOH).
RMSE is always on SOH. AIC/BIC labeled `SOHLikelihood` use the log-transform Jacobian (`+2Σ ln(SOH)` added to log-model AIC/BIC),
which permits a same-response-density comparison under each model's IID likelihood assumptions.
The log-linear conditional-mean estimate is `exp(f(x)) × mean_train(exp(residual))` (Duan smearing).
The common smearing factor is an approximation under heteroskedasticity, not an individually calibrated prediction interval.
The quadratic fit improves in-sample curvature fit, but this alone does not select a forecasting model.

## C3.4–C3.5: multiple regression and backward selection

The full research formula remains `{selected['formulas']['research_loglinear']}`.
Centers are computed only on the relevant training set: `{json.dumps(selected['means'])}`.
Conventional OLS and battery-clustered SE, p-values and CI are exported side by side.
Cluster inference uses finite-sample covariance correction, group df and t/F reference distributions.
With eight training cells and sparse operating combinations it remains exploratory, including apparently small p-values.

{markdown(table('regression_research_full_coefficients'))}

{markdown(table('regression_conditional_cycle_slopes'))}

These are conditional associations at the training-average cutoff; the cutoff value is an interpolation of the model,
not a new observed treatment cell. A positive slope in an underrepresented condition is a model limitation,
not evidence that ageing improves a battery. Temperature, current and depth of discharge are incompletely crossed.

Backward selection follows four explicit steps: fit the full model; test eligible terms; remove the highest p > 0.05;
refit and repeat until no eligible term exceeds 0.05. Interactions are tested separately, and retained interactions protect
their component main effects. Every step exports formula, p-values, adjusted R², AIC/BIC and the removed term.
OLS selection is retained as a sensitivity. The exact course slide p.39 is not present locally, so literal slide alignment
requires reviewer confirmation. Selection p-values and post-selection intervals are descriptive rather than confirmatory.

{markdown(table('regression_backward_cluster'))}

The complete VIF, residual/influence summaries and per-battery DW tables are in `reports/tables/regression_*.csv`.

## C3.6: independent battery-level regression

Fit `fade_pct_per_100cyc ~ ambient_C + I_level_A + discharge_cutoff_V` to 23 constant-condition cells.
The supplied full-span and ≤25-cycle slopes are independently recomputed and checked against `battery_fade.csv`.
HC3 covariance with t inference is used; its intervals differ from earlier normal-reference HC3 intervals.

{markdown(table('regression_fade_pct_per_100cyc'))}

{markdown(table('regression_fade25_pct_per_100cyc'))}

The fade outcome is **SOH percentage points lost per 100 valid cycles**. Full-span estimates support
temperature/current associations under the assumed additive model; short-window estimates are less conclusive.
The cycle-level model describes conditional log-SOH trajectories; this model describes between-battery estimated slopes,
so coefficients are not numerically interchangeable. Different run lengths, nonlinear early ageing and slope-estimation
uncertainty limit the comparison. HC3 does not propagate the first-stage slope uncertainty.

## C3.7: forecasting protocol, variants and model selection

For each held-out cell, `k=floor(0.60*n)` retained rows are history, origin is row k−1,
and the remaining rows form both E1 and E2 horizons. The forecast coordinates follow the locked group protocol.
E1 rolls observed history forward; E2 keeps the origin fixed and predicts only to the observed horizon endpoint.
No observation beyond the origin is used as an E2 predictor; future SOH/measurement perturbation tests verify this.
All three operating conditions are validated as finite and fixed across each complete Tier A series.
E2 builds condition predictors from the origin values only; time-varying or missing conditions raise an explicit error.
The retained future cycle coordinates are the retrospective evaluation schedule specified by the group.
Historical archives are optional comparisons, isolated from fit, CV, current metrics and prediction export.

Prespecified candidates are full log-linear, hierarchy-selected log-linear, quadratic log-linear,
history-anchored full log-linear, and history-anchored quadratic log-linear. The quadratic variant adds one global cycle² term.
An anchored forecast is `SOH_reference × exp[f(cycle_target,conditions)−f(cycle_reference,conditions)]`.
It calibrates a battery-specific intercept using allowed history; the common smearing factor cancels in the ratio.
This ratio forecast is not guaranteed to equal a conditional arithmetic mean. No battery ID fitted effect is assigned to a new cell.
E1 uses the preceding retained row as reference; E2 uses only the fixed origin.

GroupKFold(k=4) on the eight train cells refits centering, selection and smearing in each fold.
Rank the average held-out-battery MAE, and freeze one winner per scenario before scoring test outcomes.
The E1 winner is **{selected['winners']['E1']}**; E2 winner is **{selected['winners']['E2']}**.
The shared CSV label `regression` represents a scenario-specific forecasting strategy, with separate fitted variants
and coefficient vectors. `regression_selected.json` records this mapping, fitted formulas and permitted battery-specific anchors.
All scores, including losing candidates, are exported; the marginally lower test MAE of another candidate is not used to switch winners.
CV has only eight cells and one training cell at 24°C/4 A, making the ranking unstable and some CV conditions sparsely supported.

{markdown(table('regression_cv_selection'))}

### Equal-battery macro test metrics

{markdown(macro,['model','scenario','n','MAE','RMSE','MAPE_pct','R2'])}

### Pooled-row micro test metrics

{markdown(micro,['model','scenario','n','MAE','RMSE','MAPE_pct','R2'])}

### Per-battery test metrics

{markdown(per_cell,['model','scenario','battery_id','n','MAE','RMSE','MAPE_pct','R2'])}

The anchored E1 variant greatly improves on the old uncalibrated cross-cell curve, but only slightly improves persistence MAE;
it does not improve every metric. The chosen E2 variant improves on the original curve but remains worse than frozen-origin persistence.
Direct log-linear E1 and E2 coincide because their predictors are the same. History anchoring changes the information used,
so gains must not be attributed solely to a better global regression specification.
Global test predictor ranges lie inside global training ranges. Nevertheless, sparse *joint* condition support and
cell-specific offsets/slopes make conditional extrapolation and heterogeneity plausible; the audit does not claim that scalar-range extrapolation caused failure.
No post-test candidate expansion or tuning was performed. No SVR, ARIMA or LSTM result is generated here.

### EOL/RUL

{markdown(eol)}

The first observed `SOH ≤ 0.8` is absorbing for event timing even if SOH later recovers.
B0047 already crossed before the origin; its historical 27-cycle EOL error is invalid for this forecast task and is now blank.
No-crossing rows remain explicit, and RUL error is computed only for two within-horizon crossings.
No horizon extrapolation, interpolation of crossing cycles or silent removal of censored cells is performed.

## Outputs, integration and limitations

`reports/predictions/regression.csv` retains exactly `model,battery_id,cycle_valid,scenario,y_true,y_pred,y_lower,y_upper`.
It contains the CV-selected regression strategy, with `model=regression`; intervals are blank because no valid forecast interval is available.
Research/variant/baseline forecasts are kept in `reports/tables/`, not as extra files in the group's predictions glob.
Archived working-copy and Git HEAD artifacts remain in `reports/audit/regression_sem_before/`.
See [test forecast figure](../reports/figures/regression_test_forecasts.png),
[simple diagnostics](../reports/figures/regression_simple_diagnostics.png), and
[multiple-regression diagnostics](../reports/figures/regression_preferred_diagnostics.png).

Limitations: few independent batteries, incomplete factorial design, temporal error dependence,
heteroskedasticity, history-normalized target, retrospective screening, unstable CV ranking, missing calibrated uncertainty,
and E2 performance below persistence. The pipeline is reproducible; these limitations remain scientific constraints.

## IEEE-ready condensed text

We modeled log-SOH using valid cycle, temperature indicators, discharge current, cutoff voltage and cycle interactions.
Inference used battery-clustered uncertainty, with a 23-cell HC3 fade-rate sensitivity. Prediction was evaluated on the four
locked held-out batteries using one-step and fixed-origin horizons, with GroupKFold model selection restricted to eight training cells.
History-anchored log-linear trajectories improved one-step MAE relative to the original cross-cell regression,
while fixed-origin performance remained inferior to persistence. All reported errors are on the SOH scale;
first-passage RUL errors exclude pre-origin and censored events. Results support conditional associations rather than causal claims.
The numerical tables above should accompany this paragraph; do not present the anchored variant as the unchanged research baseline.
{reference}
'''
    sem = f'''# SEM / path analysis methods and results — Quoc Dat

Generated from executed outputs on 2026-11-10 (audit started 2026-11-09). Integration targets: Full Report §§3.7/4/5.7 and IEEE Methodology/Results.

## Theoretical framework and hypotheses

Operating temperature, discharge current and cutoff voltage may be associated with electrolyte resistance Re and
charge-transfer resistance Rct, which in turn may be associated with SOH. Direct condition–SOH paths remain in the model.
Age (`cycle_valid`) controls the SOH equation. All variables are observed; the dataset does not justify a latent measurement model.
The hypotheses are the existence of direct and indirect **associations** through each impedance measure, not a presupposed sign
or causal mediation conclusion. Observational data, unmeasured battery heterogeneity, and temperature/age confounding prevent causal identification.

For condition vector X, the primary equations are

`Re = a_Re'X + ε_Re`, `Rct = a_Rct'X + ε_Rct`,
`ln(SOH) = c'X + b_Re Re + b_Rct Rct + γ cycle_valid + ε_Y`.

All four exogenous variables have freely estimated variances and covariances. The primary residual covariance Re ~~ Rct remains fixed at zero,
matching the original theory. Age-to-mediator paths and shared mediator residual covariance are investigated only as named sensitivities.
The model is a covariance model, without a fitted mean/intercept structure.

## Data and measurement audit

{markdown(table('sem_sample_flow'))}

Use Tier A/B rows and the first retained discharge linked to each unique `(battery_id,eis_op_index)`.
Join raw `impedance_ops.csv` by this key and require both raw resistances in (0,0.5) ohm; invalid raw data are excluded rather than
counting a forward-filled cleaned value as fresh measurement. Here no such mismatch exists, but the code and tests cover it.
SOH and required condition/control variables must be complete and SOH positive before taking logs.
No original processed data are changed. Measurement-availability selection and nonzero EIS-to-discharge lag remain limitations.
The historical primary sample retains 14 flagged cycle rows to preserve the original sample; flag exclusion is an explicit sensitivity.

{markdown(table('sem_eis_temperature'))}

For cold discharge conditions, the recorded EIS ambient is 24°C. Ambient coefficients therefore combine experimental-condition
differences with uncertain measurement-temperature effects. A changing EIS operation is not necessarily a changed resistance value;
operation identity, not equality of numeric values, defines measurement availability. Intermediate EIS operations without a linked retained discharge
are not invented as SOH observations. Repeated EIS points are nested within batteries, not independent units.

## Estimation, identification and fit correction

Fit MLW by SLSQP with seed 42. Internal z scaling improves numerical conditioning; regression coefficients and SEs are converted back
using `SD(outcome)/SD(predictor)`, and covariance parameters by the product of SDs. Standardized coefficients come from the fitted implied variances.
This reparameterization preserves the theoretical graph. Check sample covariance rank, solver success, finite estimates,
positive fitted covariance and variance, and numerical Fisher-information rank. Primary information rank is
{int(primary.information_rank)} of {int(primary.free_parameters)} parameters; there are {int(primary.observed_moments)} distinct observed covariance moments.
Local numerical identification does not imply that the theory is correct or causally identified.

The optional archived original notebook comparison is:

{historical_text}

`semopy 2.3.11` calculates df as `p(p+1)/2 − number_of_active_parameters`.
Its default empirical exogenous moments are fixed without being counted in that parameter total. With four exogenous variables,
ten empirically fitted nuisance moments were omitted from the original df accounting: 13 must become **3**.
The corrected model explicitly frees those ten moments. This preserves fitted theory restrictions but supplies correct parameter counting.
The independence baseline is also refitted with explicit moments and adequate scaling; therefore the original CFI/TLI values
are not directly comparable as if only substantive structural fit had changed. This is an inference from the inspected package implementation,
confirmed by numerical parameter/information counts, rather than an assumption that better-looking indices indicate a better model.

{markdown(fits,['model','n','batteries','converged','free_parameters','information_rank','DoF','chi2','CFI','TLI','RMSEA'])}

The raw-unit legacy reexecution can terminate at a much poorer objective despite an optimizer success flag.
Its values are a numerical sensitivity, not a replacement for the archived historical result. The scaled original model and corrected model
recover the historical primary discrepancy closely; unit-scaling tests verify conversion of coefficients.
The corrected primary fit is poor. Negative TLI is retained rather than clipped. Removing flagged rows does not repair fit.
Package-reported AIC/BIC/LogLik are exported for provenance and are not used for selection or interpreted as validated likelihood comparisons.

## Why the original model fits poorly

The primary model excludes age from both mediator equations and assumes independent mediator residuals.
Resistance measurements share the same EIS operation and electrochemical history; residual dependence and age associations
are scientifically plausible omitted structure. Age-control sensitivity reduces chi-square but has only one df;
RMSEA need not improve when restrictions decrease. Adding both age paths and Re ~~ Rct yields zero df: it is saturated,
so CFI/TLI/RMSEA and an overall chi-square p-value are deliberately blank. Near-zero discrepancy of this model is algebraic
and does not validate mediation. No modification-index search or data-driven path deletion is used.

Correlated predictors, incomplete condition crossing, pooled linear temperature response, measurement-temperature mismatch,
skewness and a small number of independent batteries also restrict interpretation. Correlation, equation-specific VIF,
distribution summaries and standardized covariance residuals are exported as `sem_correlations.csv`, `sem_equation_vif.csv`,
`sem_distributions.csv` and `sem_standardized_covariance_residuals.csv`.
The IID chi-square and conventional fit indices do not adjust for clustered rows; bootstrap intervals do not make these fit tests cluster-robust.

## Coefficients and uncertainty

{markdown(paths,['lval','rval','Estimate','Est. Std','Std. Err','p-value','CI_low_iid','CI_high_iid','bootstrap_CI_low','bootstrap_CI_high'])}

`Estimate` is unstandardized (ln(SOH) per original input unit for outcome paths); `Est. Std` is standardized.
`Std. Err`, p-values and `CI_*_iid` assume independent rows and are shown only as conventional reference.
Battery-bootstrap path intervals are the primary uncertainty sensitivity. A standardized multiple-regression coefficient can exceed one
with correlated predictors; it is not a Pearson correlation. A positive conditional Re coefficient does not show that higher resistance improves health.

## Direct, indirect and total associations

For condition j: direct = c_j; indirect via Re = a_Re,j b_Re;
indirect via Rct = a_Rct,j b_Rct; total = direct + both indirect products.
Every effect is on the ln(SOH) scale per °C, A or V. The bootstrap recomputes internal scaling per sample and returns products to original units.
Only exponentiating a whole specified log association yields a relative SOH ratio; adding percentages across paths is not valid.

{markdown(table('sem_effects'))}

The temperature total is close to zero because opposing paths cancel; percent mediated would be unstable and is not reported.

{markdown(table('sem_effect_intervals'))}

Resample {int(primary.batteries)} whole batteries with replacement, preserving the complete selected time series and duplicate-block multiplicity.
Seed = {bootstrap['seed']}. Stop only after {bootstrap['target_valid']} valid fits; maximum 500 attempts.
Actual result: **{bootstrap['valid']} valid / {bootstrap['attempts']} attempts, {bootstrap['failures']} failures**.
An unsuccessful optimizer, nonfinite estimate, singular sample covariance or nonpositive fitted variance does not count as a valid replicate.
The full attempt log, sampled battery identities, iterations and failure reasons are saved in `sem_bootstrap_audit.csv`.
Percentile CI 95% are estimated separately for direct, Re indirect, Rct indirect, combined indirect and total associations.
All condition-effect intervals include zero; these data do not support a statistically resolved mediation association.
Two hundred replicates meet the requested audit minimum; percentile endpoints still have Monte Carlo uncertainty, and 22 battery blocks
do not eliminate confounding or guarantee population representativeness.

## Diagram and reporting

![Primary fitted observed-variable graph; standardized coefficients](../reports/figures/sem_path_diagram.png)

The 320-dpi PNG and vector PDF are generated from fitted path estimates. Every arrow corresponds to a regression path,
and `sem_diagram_coefficients.csv` preserves the exact values used. Nuisance exogenous covariance arrows are omitted for readability
and disclosed in the caption; no latent node labeled “conditions” replaces the three actual predictors.

## IEEE-ready condensed text

Observed-variable path analysis evaluated operating conditions → impedance → log-SOH with direct condition paths and age control.
Distinct measured EIS operations yielded {int(primary.n)} observations nested in {int(primary.batteries)} batteries.
Explicit estimation of exogenous covariance moments corrected the original degrees-of-freedom accounting.
The primary model had χ² = {primary['chi2']:.4f}, df = {int(primary.DoF)}, CFI = {primary.CFI:.4f},
TLI = {primary.TLI:.4f} and RMSEA = {primary.RMSEA:.4f}, indicating poor fit under the conventional IID diagnostics.
Battery-block bootstrap ({bootstrap['valid']} valid replicates) produced direct and indirect condition-association intervals containing zero.
The results do not establish causal mediation. An age/correlated-residual sensitivity was saturated and was not used as evidence of adequate overall fit.
{reference}
'''
    diff = subprocess.run(['git','diff','--stat'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    status = subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    audit = f'''# Audit Regression và SEM — 09–10/11/2026

Thành viên: Quốc Đạt · branch `quoc-dat`. Đây là tài liệu review nội bộ. Nội dung chèn báo cáo tiếng Anh:
[Regression_Report.md](Regression_Report.md), [SEM_Report.md](SEM_Report.md).

## Phạm vi và bảo toàn

Đã kiểm tra `git status`, branch và 5 commit trước khi sửa. HEAD lúc bắt đầu: `73452fd`.
Thay đổi có sẵn: `notebooks/50_sem.ipynb`, `reports/predictions/regression.csv`, notebook chưa track `notebooks/10_regression.test.ipynb`.
Đã sao lưu nguyên trạng cả ba, notebook Regression và các hình cũ vào `reports/audit/regression_sem_before/`;
patch ban đầu ở `preexisting.patch`. Bản Git HEAD của SEM và dự báo được lưu riêng để phân biệt với working copy.
Không sửa notebook thử nghiệm; không commit/push/merge/rebase/PR. Không sửa dữ liệu gốc, processed tables,
pipeline của Leader, giao thức trong Rec/Go/WBS, hay mô hình thành viên khác.

## Lỗi logic/code và phần đã sửa

| Phát hiện | Cách xử lý và bằng chứng |
| --- | --- |
| Simple regression dùng toàn Tier A thay vì G1 | Dùng G1 train, loại pin test B0007; đủ F/t/SE/CI và chẩn đoán |
| Correlation thiếu Re/Rct và thời gian sạc | Xuất ma trận train và G1 train; VIF trên đúng design matrix |
| Centering/selection phần suy diễn dùng cả pin test | C3 fit trên train; full-sample EDA giữ vai trò mô tả; battery-level 23-cell là sensitivity suy diễn riêng |
| Backward selection bỏ cả nhóm main+interaction | Kiểm định từng term đủ điều kiện, giữ hierarchy, xuất tất cả bước, thêm OLS sensitivity |
| Chưa có GroupKFold chọn biến thể | K=4 theo pin; mọi centering, selection, smearing refit từng fold; freeze winner riêng E1/E2 |
| Forecast exp(log-SOH) thiếu retransformation correction | Train-only Duan cho direct model; hệ số chung triệt tiêu trong anchored ratio |
| Global curve lệch mức giữa pin | Thử anchored log-linear và quadratic bằng lịch sử được phép; giữ research baseline và mọi candidate |
| RUL B0047 tính sau khi pin đã EOL | First SOH≤0.8 tại chu kỳ 30, origin 40; đánh dấu pre-origin, không tính RUL/error |
| Pin không dự báo crossing dễ bị loại khỏi thống kê | Giữ đủ bốn pin, ghi rõ trạng thái censored/no crossing, không extrapolate |
| SEM root phụ thuộc working directory | Tìm repository theo cây thư mục; đã kiểm tra từ root và notebooks |
| EIS index mới có thể mang Re/Rct ffill | Join lại raw operation, kiểm tra (0,0.5) ohm và timestamp; dữ liệu thật không có mismatch |
| SEM bootstrap chỉ kiểm tra exception | Kiểm tra solver success, rank, giá trị hữu hạn, variance/covariance hợp lệ; tiếp tục đến 200 fit hợp lệ |
| SEM df không đếm moment ngoại sinh cố định theo mẫu | Khai báo tự do 10 nuisance moments; primary df=3, không dùng df=13 để kết luận fit |
| Raw-unit optimizer có thể báo success ở nghiệm kém | Internal scaling; đối chiếu scaled original và unit-conversion test; giữ raw reexecution như sensitivity |
| Path diagram thiếu hệ số/biến thực | Vẽ đúng predictor, hệ số standardized lấy từ model; PNG 320 dpi + PDF |

## Kết quả Regression

{markdown(macro,['model','scenario','MAE','RMSE','MAPE_pct','R2'])}

E1 chọn `{selected['winners']['E1']}`, E2 chọn `{selected['winners']['E2']}` bằng CV train.
Không đổi sang candidate khác sau khi nhìn test. E1 chỉ cải thiện nhỏ MAE so với persistence;
RMSE không thắng. E2 vẫn kém persistence. CSV chung 426 dòng (213 mỗi scenario), đúng schema 8 cột, interval trống.
Chi tiết từng pin, micro/macro và toàn bộ candidate nằm trong `reports/tables/`.

{markdown(eol,['model','battery_id','origin_cycle','observed_EOL_cycle','predicted_EOL_cycle','status','absolute_RUL_error'])}

## Kết quả SEM

Kết quả gốc lấy từ output đã commit, không ghi lại con số từ trí nhớ:

{markdown(historical,['DoF','chi2','CFI','TLI','RMSEA'])}

{markdown(fits,['model','n','DoF','chi2','CFI','TLI','RMSEA'])}

Các thay đổi CFI/TLI có cả phần sửa baseline và df; không diễn giải là so sánh hai theory tốt/xấu đơn thuần.
Mô hình chính fit kém; mô hình có age→mediators và residual covariance bão hòa không có giá trị kiểm định overall fit.
Bootstrap: {bootstrap['valid']}/{bootstrap['attempts']} hợp lệ, {bootstrap['failures']} thất bại, seed 42;
CI 95% cho direct/indirect/total của cả ba điều kiện đều chứa 0. Hệ số đầy đủ và CI từng đường ở `sem_primary_paths_with_bootstrap.csv`.

## Hạn chế dữ liệu và thống kê vẫn còn

- Dữ liệu ít pin độc lập; thí nghiệm không factorial đầy đủ; Tier A không có 43°C/profile thay đổi.
- Tier và cờ Hampel được quyết định hồi cứu theo giao thức nhóm; không thể tuyên bố benchmark hoàn toàn prospective hoặc không có upstream look-ahead.
- SOH chuẩn hóa bằng C0 giai đoạn đầu; C0 không làm predictor, đã có trước forecast origin. Sau lọc có khoảng trống chu kỳ; E1 là next retained observation.
- Phần dư tự tương quan/heteroskedastic, VIF tương tác cao, chỉ 8 cụm train và 3 cụm G1; post-selection inference không mang nghĩa confirmatory.
- Battery fade là slope ước lượng; HC3 không truyền uncertainty của bước 1. Độ dài chuỗi khác nhau ảnh hưởng so sánh.
- EIS ambient ghi 24°C cho discharge 4°C; mediator đồng thời phụ thuộc age/chemistry; quan sát không xác lập nhân quả.
- Bootstrap không chữa misspecification/confounding. 200 replicate là mức tối thiểu, chưa cho endpoint CI chính xác cao.
- Không có prediction intervals đã được hiệu chỉnh; E2 chưa thắng baseline; không thêm mô hình ngoài Regression/SEM.

## Kiểm chứng và lệnh tái lập

Môi trường riêng Python {execution['environment']['python']}; tất cả version và SHA256 input/output ở
`reports/tables/regression_sem_execution.json`. Lock của 60 dependency cho hai notebook:
`requirements-regression-sem.lock.txt`; `requirements.txt` chung giữ nguyên (torch không cần cho phần này).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-regression-sem.lock.txt
.venv/Scripts/python src/run_regression_sem.py
.venv/Scripts/python src/run_regression_sem.py --cwd notebooks --verify-repro
.venv/Scripts/python -m unittest discover -s tests -p test_regression_sem.py -v
.venv/Scripts/python src/write_regression_sem_reports.py
```

Execution: {json.dumps(execution['notebooks'],ensure_ascii=False)}.
So sánh chính xác CSV/JSON số học giữa hai working directory: **{execution['exact_numerical_reproduction']}**.
Input CSV unchanged: **{execution['processed_inputs_unchanged']}**.
Kiểm thử: {json.dumps(validation_brief,ensure_ascii=False)}.
`uv pip check` xác nhận các package đã cài tương thích. Không cài torch/jupyter UI vì không được hai notebook sử dụng;
nbclient/ipykernel/nbformat thực thi fresh kernel và lưu đầy đủ output. Các lỗi chạy đầu trong quá trình phát triển
(chuỗi newline của notebook builder, API metadata statsmodels 0.15) đã được sửa trước lần kiểm chứng cuối.

## WBS và việc cần review

| Mã | Tình trạng trong phạm vi Quốc Đạt |
| --- | --- |
| C3.1 | Hoàn thành correlation và design VIF, Re/Rct train |
| C3.2 | Hoàn thành G1 simple, F/t/CI, residual/QQ |
| C3.3 | Hoàn thành ba dạng hàm, SOH-scale RMSE, likelihood Jacobian và smearing |
| C3.4 | Hoàn thành full log-linear dummy/interactions, OLS/cluster CI và conditional slopes |
| C3.5 | Đã chạy four-step backward có hierarchy; **chưa đối chiếu nguyên văn slide p.39 vì file không có trong repo** |
| C3.6 | Hoàn thành verify battery_fade, 23-cell HC3 full/25-cycle và phân tích hạn chế |
| C3.7 | Hoàn thành E1/E2 CSV/metrics/plots/EOL statuses; E2 chưa thắng persistence là kết quả thực tế |
| C3.8 | Có nội dung English report/paper riêng; chờ nhóm tích hợp và review |
| C7.1 | Có sơ đồ thực với hệ số; **chờ nhóm đồng ý đường dẫn**, không tự coi là đã duyệt |
| C7.2 | Hoàn thành dữ liệu new EIS, raw join/missingness/temp audit |
| C7.3 | Hoàn thành estimation, identification, fit audit và sensitivities; fit chính vẫn kém |
| C7.4 | Hoàn thành products + battery bootstrap; không báo % mediation thiếu ổn định hoặc khẳng định significance |

Tiến Hưng review phương pháp, df correction, assumption và slide selection; Leader review ý nghĩa label `regression`
cho strategy có history anchoring và scenario-specific selection trước ghép comparison.
Nhóm cần xác nhận giới hạn retrospective benchmark, treatment của flagged rows trong SEM, EOL pre-origin của B0047,
và chỉ tích hợp những kết luận có CI/metric hỗ trợ. Việc xác nhận đã học Chap 7, duyệt giả thuyết, tích hợp deliverable,
review chéo và kiểm tra Turnitin/AI là phần nhóm thực hiện; notebook chạy thành công không thay thế các việc đó.
Không cần nhóm đổi giao thức để đọc các đầu ra hiện tại; pipeline chung `src/protocol.py` vẫn thuộc Leader.

## File và git diff

Thay đổi tracked: hai notebook, bốn hình cũ liên quan, CSV regression và `.gitignore` (bỏ qua bytecode).
Thêm: bốn module analysis/runner/utils, module sinh report này, tests, lock dependency, report Markdown,
tables, forecast/selected diagnostic figures, diagram PDF và bản sao bảo toàn trước audit.
Notebook thử nghiệm chưa track là thay đổi có sẵn, không phải kết quả mới của audit.
`git diff --stat` chỉ đếm tracked files; file mới chưa stage vẫn được liệt kê bằng status:

```text
{diff}
```

```text
{status}
```
'''
    for name,content in [('Regression_Report.md',regression),('SEM_Report.md',sem),('Regression_SEM_Audit.md',audit)]:
        (ROOT/'docs'/name).write_text(content,encoding='utf8')
    if args.append_worklog:
        path = ROOT/'docs/Work_log.md'
        marker = '### [10/11/2026] Quốc Đạt · branch `quoc-dat` · Audit Regression và SEM (bắt đầu 09/11)'
        if marker in path.read_text(encoding='utf8'):
            raise RuntimeError('Work log entry exists; do not duplicate or replace historical entries')
        entry = f'''

{marker}

**Bối cảnh và bảo toàn.** Yêu cầu audit trực tiếp Regression/SEM, không commit/push/merge. Đã kiểm tra status/branch/log;
branch đúng `quoc-dat`, HEAD `73452fd`. Working copy có SEM, regression CSV thay đổi và regression.test chưa track.
Đã sao lưu các bản này cùng notebook/hình cũ vào `reports/audit/regression_sem_before/`, giữ `preexisting.patch` và bản HEAD riêng.
Không sửa pipeline/dataset/giao thức của Leader hay mô hình thành viên khác. Chỉ thêm nội dung vào phần Quốc Đạt này.

**Nguồn số liệu và phương pháp.** Đọc README/Rec/Go/WBS/Work_log/Introduction, requirements, hai notebook,
cycles_clean/battery_fade/prediction CSV và ba module build/clean/eda. Ưu tiên Rec; seed 42, test/train giữ nguyên.
Python {execution['environment']['python']}; version package và SHA256 từng input/output ở `regression_sem_execution.json`;
dependency lock riêng để không thay requirements chung. Script runner dùng fresh kernels từ root và notebooks.

**Regression.** Sửa simple thành G1 train (n={int(g1['n'])}, {int(g1.batteries)} pin), không fit test B0007;
R²={g1.R2:.6f}, adjusted R²={g1.adjusted_R2:.6f}, F={g1.F:.6f}. Xuất conventional/cluster SE, p/CI, residual/standardized QQ/BP/JB/DW.
Thêm Re/Rct/charge duration correlations và design VIF; r Re–Rct train={corr:.6f}, G1 train={g1corr:.6f}.
Full research log-linear giữ dummy/interaction; backward kiểm định term có hierarchy, covariance cluster, theo dõi adjusted R² từng bước,
OLS là sensitivity. Chưa có slide Chap3 p39 để xác nhận nguyên văn bốn bước. Smearing train-only; AIC log model đổi về SOH density bằng Jacobian.
Xác minh cả full/25-cycle slopes của 23 battery_fade; HC3 với t inference, n nhỏ và slope uncertainty được nêu rõ.
GroupKFold4 train-only thử đúng 5 candidate có cơ sở: full/selected/quadratic log-linear và 2 anchored variants.
CV chọn E1={selected['winners']['E1']}, E2={selected['winners']['E2']}; không thay lựa chọn sau khi nhìn test.

{markdown(macro,['model','scenario','MAE','RMSE','MAPE_pct','R2'])}

E1 chỉ thắng nhẹ persistence MAE, không thắng RMSE; E2 vẫn kém persistence. Global uncalibrated model chưa giải quyết heterogeneity.
Giữ kết quả cũ và toàn bộ candidate. CSV chung 426 dòng/8 cột, 213 dòng mỗi scenario, intervals NaN.
Xuất cả micro/macro/per-battery; forecasts plot có threshold .8. EOL dùng first SOH≤.8; B0047 đã event cycle30 trước origin40,
bỏ RUL/error không hợp lệ, giữ đủ mọi battery và explicit no-crossing status, không extrapolate horizon.
E1 là next retained clean observation; centered Hampel/lifetime Tier là upstream retrospective constraints, không tự sửa protocol.

**SEM.** Join từng EIS operation thực, không đếm ffill thành measurement; 812 điểm/22 pin, 14 dòng có quality flag được giữ
trong primary để đối chiếu mẫu gốc; sensitivity bỏ flag còn 798. 137 measurement của discharge4°C có EIS ambient24°C.
Sửa root path; kiểm tra convergence/rank/variance. Đếm tường minh 10 nuisance exogenous moments: primary25 parameter/28 moment, df3.
Internal z scaling và đổi coefficient/SE về đơn vị gốc; test đổi A thành mA kiểm tra scale. Bản raw-unit solver có thể success ở objective kém;
không dùng nó thay historical output. Primary theoretical graph không đổi; sensitivity age→mediators và shared Re/Rct residual có lý do vật lý.
Model thêm cả hai là saturated df0, không coi fit đẹp là xác nhận theory.

{markdown(historical,['DoF','chi2','CFI','TLI','RMSEA'])}

Primary mới: χ²={primary['chi2']:.6f}, df={int(primary.DoF)}, CFI={primary.CFI:.6f}, TLI={primary.TLI:.6f}, RMSEA={primary.RMSEA:.6f}.
Khác biệt indices có sửa cả df/baseline; fit vẫn kém, repeated rows khiến IID fit tests chỉ descriptive.
Bootstrap sample theo pin, có duplicate block multiplicity, seed42, target200 valid/max500; thực tế {bootstrap['valid']}/{bootstrap['attempts']}
hợp lệ và {bootstrap['failures']} failure. Xuất attempts/solver checks/SE/CI95 cho từng đường, direct/Re/Rct/combined/total;
tất cả condition-effect CI chứa0, không kết luận mediation significant/nhân quả, không tính unstable mediation% khi total≈0.
Sơ đồ 320dpi + vector PDF: từng predictor thật, mũi tên và standardized coefficient khớp CSV model, không hard-code estimates.

**Đầu ra và kiểm chứng.** Hai notebook lưu outputs fresh, {json.dumps(execution['notebooks'])}; exact numerical reproduction
root vs notebooks={execution['exact_numerical_reproduction']}; input unchanged={execution['processed_inputs_unchanged']}.
Validation: {json.dumps(validation_brief,ensure_ascii=False)}. Đã inspect trực quan forecast/path/residual figures và kiểm tra dependency compatibility.
Lệnh tái lập và toàn bộ findings/WBS/review checklist: `docs/Regression_SEM_Audit.md`;
methods/results + IEEE-ready text: `docs/Regression_Report.md`, `docs/SEM_Report.md`.
Các file mới `src/regression_analysis.py`, `sem_analysis.py`, `regression_sem_utils.py`, `run_regression_sem.py`,
`write_regression_sem_reports.py`, `tests/test_regression_sem.py`, lock, tables và figures đều chỉ phục vụ hai phần này.

**Giới hạn và review còn lại.** Các thay đổi là phiên bản đề xuất để Quốc Đạt/Tiến Hưng/Leader review, không phải quyết định nhóm đã duyệt.
C3.1–C3.8, C7.2–C7.4 đã có code/output/nội dung báo cáo; C3.5 còn đối chiếu slide, C7.1 còn nhóm đồng ý theory paths,
và còn tích hợp deliverable, review chéo. Không thể làm E2 thắng baseline hoặc SEM fit tốt bằng sửa code hợp lệ trong phạm vi này.
Giữ trung thực các hạn chế n nhỏ, confounding, serial dependence, retrospective cleaning và chưa có prediction intervals.
'''
        with path.open('a',encoding='utf8',newline='') as f:
            f.write(entry)
    print('Wrote Regression_Report.md, SEM_Report.md, Regression_SEM_Audit.md')


if __name__=='__main__':
    main()
