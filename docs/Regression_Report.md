# Regression methods and results — Quoc Dat

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
Training cells: B0005, B0006, B0018, B0033, B0045, B0046, B0048, B0056. Test cells: B0007, B0034, B0047, B0055. There is no battery overlap.
The G1 simple-regression sample has 465 rows from 3 training batteries;
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

Train Re–Rct Pearson r = 0.510722; G1 train r = 0.953088. These differ from published full-sample EDA values because the sample differs.
Pairwise correlation uses available pairs; VIF uses the actual complete-case dummy/interaction design.

| term | VIF |
| --- | --- |
| Intercept | 1 |
| C(ambient_C)[T.24.0] | 2.58086 |
| cycle_c | 14.0496 |
| cycle_c:C(ambient_C)[T.24.0] | 13.906 |
| current_c | 3.83272 |
| cutoff_c | 2.87222 |
| cycle_c:current_c | 6.05296 |
| cycle_c:cutoff_c | 4.74358 |

Centering reduces nonessential interaction collinearity but does not eliminate missing condition combinations.
The supplementary EIS design (`regression_eis_design_vif.csv`) has substantial Rct/condition collinearity;
both impedance predictors are therefore not added to the primary regression. A threshold of 10 is a diagnostic,
not an automatic argument for deleting a theoretically required main effect.

## C3.2–C3.3: G1 simple regression and functional forms

| n | batteries | R2 | adjusted_R2 | F | F_p | BP_LM | BP_p | JB | JB_p | max_leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 465 | 3 | 0.861688 | 0.86139 | 2884.51 | 5.12777e-201 | 48.3671 | 3.53455e-12 | 19.8607 | 4.86753e-05 | 0.0099082 |

| term | coefficient | OLS_SE | OLS_p | OLS_CI_low | OLS_CI_high | robust_type | robust_SE | robust_t | robust_p | robust_CI_low | robust_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | 0.997255 | 0.00385005 | 0 | 0.989689 | 1.00482 | cluster | 0.0122414 | 81.4655 | 0.000150645 | 0.944584 | 1.04993 |
| cycle_valid | -0.0022442 | 4.17855e-05 | 5.12777e-201 | -0.00232632 | -0.00216209 | cluster | 0.000194277 | -11.5516 | 0.00741089 | -0.00308011 | -0.0014083 |

OLS F/t tests assume independent errors; the displayed battery-clustered sensitivity has only three independent groups
and should not support strong population claims. Residual plots, standardized Q-Q plots, Breusch–Pagan,
Jarque–Bera and battery-specific Durbin–Watson diagnostics are saved. Heteroskedasticity and serial dependence remain.

| model | response | n | R2_response | adjusted_R2_response | smearing | RMSE_SOH | AIC_SOHLikelihood | BIC_SOHLikelihood |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | SOH | 465 | 0.861688 | 0.86139 | 1 | 0.0419042 | -1626.69 | -1618.41 |
| loglinear | ln(SOH) | 465 | 0.852313 | 0.851994 | 1.00145 | 0.0412416 | -1587.94 | -1579.65 |
| quadratic | SOH | 465 | 0.866281 | 0.865702 | 1 | 0.0412026 | -1640.39 | -1627.97 |

R² and adjusted R² are on the stated response scale and must not be ranked across SOH versus ln(SOH).
RMSE is always on SOH. AIC/BIC labeled `SOHLikelihood` use the log-transform Jacobian (`+2Σ ln(SOH)` added to log-model AIC/BIC),
which permits a same-response-density comparison under each model's IID likelihood assumptions.
The log-linear conditional-mean estimate is `exp(f(x)) × mean_train(exp(residual))` (Duan smearing).
The common smearing factor is an approximation under heteroskedasticity, not an individually calibrated prediction interval.
The quadratic fit improves in-sample curvature fit, but this alone does not select a forecasting model.

## C3.4–C3.5: multiple regression and backward selection

The full research formula remains `ln_soh ~ cycle_c + C(ambient_C) + current_c + cutoff_c + cycle_c:C(ambient_C) + cycle_c:current_c + cycle_c:cutoff_c`.
Centers are computed only on the relevant training set: `{"cycle_valid": 70.63569165786694, "I_level_A": 2.1615628299894403, "discharge_cutoff_V": 2.4200633579725452}`.
Conventional OLS and battery-clustered SE, p-values and CI are exported side by side.
Cluster inference uses finite-sample covariance correction, group df and t/F reference distributions.
With eight training cells and sparse operating combinations it remains exploratory, including apparently small p-values.

| term | coefficient | OLS_SE | OLS_p | OLS_CI_low | OLS_CI_high | robust_type | robust_SE | robust_t | robust_p | robust_CI_low | robust_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | -0.162658 | 0.00350621 | 3.54241e-245 | -0.169538 | -0.155777 | cluster | 0.0123536 | -13.1668 | 3.40333e-06 | -0.191869 | -0.133446 |
| C(ambient_C)[T.24.0] | -0.0377553 | 0.00406882 | 1.1472e-19 | -0.0457403 | -0.0297702 | cluster | 0.0231553 | -1.63052 | 0.147012 | -0.0925088 | 0.0169983 |
| cycle_c | -0.00138559 | 9.18337e-05 | 3.21219e-46 | -0.00156581 | -0.00120536 | cluster | 0.00022802 | -6.07658 | 0.000502561 | -0.00192477 | -0.000846403 |
| cycle_c:C(ambient_C)[T.24.0] | -0.00164743 | 0.000105525 | 5.05996e-49 | -0.00185452 | -0.00144033 | cluster | 0.000347038 | -4.74711 | 0.00209052 | -0.00246804 | -0.000826813 |
| current_c | 0.104701 | 0.0023875 | 1.78276e-229 | 0.100015 | 0.109386 | cluster | 0.0150957 | 6.93578 | 0.000224025 | 0.0690049 | 0.140396 |
| cutoff_c | 0.21287 | 0.00713232 | 3.48323e-138 | 0.198873 | 0.226867 | cluster | 0.0422835 | 5.03436 | 0.00150599 | 0.112886 | 0.312855 |
| cycle_c:current_c | 0.0017867 | 5.64068e-05 | 2.28455e-150 | 0.001676 | 0.00189739 | cluster | 0.000324377 | 5.50809 | 0.000898808 | 0.00101967 | 0.00255372 |
| cycle_c:cutoff_c | 0.00320548 | 0.000176103 | 1.21586e-63 | 0.00285988 | 0.00355108 | cluster | 0.000938976 | 3.41381 | 0.0112296 | 0.000985155 | 0.0054258 |

| ambient_C | I_level_A | cutoff_V | log_slope | relative_pct_per_cycle | CI_low_pct | CI_high_pct | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 2.42006 | -0.00346094 | -0.345496 | -0.440497 | -0.250404 | 5.81199e-05 |
| 4 | 2 | 2.42006 | -0.00167425 | -0.167285 | -0.220263 | -0.114279 | 0.000142202 |
| 24 | 2 | 2.42006 | -0.00332168 | -0.331617 | -0.393948 | -0.269246 | 4.69019e-06 |
| 24 | 4 | 2.42006 | 0.000251715 | 0.0251747 | -0.0679982 | 0.118434 | 0.54335 |

These are conditional associations at the training-average cutoff; the cutoff value is an interpolation of the model,
not a new observed treatment cell. A positive slope in an underrepresented condition is a model limitation,
not evidence that ageing improves a battery. Temperature, current and depth of discharge are incompletely crossed.

Backward selection follows four explicit steps: fit the full model; test eligible terms; remove the highest p > 0.05;
refit and repeat until no eligible term exceeds 0.05. Interactions are tested separately, and retained interactions protect
their component main effects. Every step exports formula, p-values, adjusted R², AIC/BIC and the removed term.
OLS selection is retained as a sensitivity. The exact course slide p.39 is not present locally, so literal slide alignment
requires reviewer confirmation. Selection p-values and post-selection intervals are descriptive rather than confirmatory.

| step | formula | covariance | adjusted_R2 | AIC | BIC | eligible_p_values | removed | removed_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ln_soh ~ cycle_c + C(ambient_C) + current_c + cutoff_c + cycle_c:C(ambient_C) + cycle_c:current_c + cycle_c:cutoff_c | cluster | 0.902978 | -3580.39 | -3541.56 | {"cycle_c:C(ambient_C)": 0.0020905197221860257, "cycle_c:current_c": 0.0008988082359386017, "cycle_c:cutoff_c": 0.011229566971938425} | none | — |

The complete VIF, residual/influence summaries and per-battery DW tables are in `reports/tables/regression_*.csv`.

## C3.6: independent battery-level regression

Fit `fade_pct_per_100cyc ~ ambient_C + I_level_A + discharge_cutoff_V` to 23 constant-condition cells.
The supplied full-span and ≤25-cycle slopes are independently recomputed and checked against `battery_fade.csv`.
HC3 covariance with t inference is used; its intervals differ from earlier normal-reference HC3 intervals.

| term | coefficient | OLS_SE | OLS_p | OLS_CI_low | OLS_CI_high | robust_type | robust_SE | robust_t | robust_p | robust_CI_low | robust_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | 55.8514 | 17.5765 | 0.0049565 | 19.0635 | 92.6393 | HC3 | 21.1114 | 2.64556 | 0.0159521 | 11.6648 | 100.038 |
| ambient_C | 0.691479 | 0.1937 | 0.00204395 | 0.28606 | 1.0969 | HC3 | 0.172658 | 4.0049 | 0.00075766 | 0.330101 | 1.05286 |
| I_level_A | -8.65212 | 2.36011 | 0.00164227 | -13.5919 | -3.71234 | HC3 | 2.20807 | -3.9184 | 0.000923182 | -13.2737 | -4.03057 |
| discharge_cutoff_V | -12.1916 | 6.70158 | 0.0846767 | -26.2181 | 1.83502 | HC3 | 7.86703 | -1.5497 | 0.13771 | -28.6574 | 4.27432 |

| term | coefficient | OLS_SE | OLS_p | OLS_CI_low | OLS_CI_high | robust_type | robust_SE | robust_t | robust_p | robust_CI_low | robust_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | 103.594 | 41.3538 | 0.0215094 | 17.0395 | 190.149 | HC3 | 60.3657 | 1.71611 | 0.102403 | -22.7528 | 229.941 |
| ambient_C | -0.0957513 | 0.455737 | 0.835825 | -1.04962 | 0.858118 | HC3 | 0.441224 | -0.217013 | 0.830511 | -1.01924 | 0.827741 |
| I_level_A | -9.68636 | 5.55286 | 0.0972469 | -21.3086 | 1.93591 | HC3 | 6.23015 | -1.55476 | 0.136504 | -22.7262 | 3.35349 |
| discharge_cutoff_V | -20.3465 | 15.7675 | 0.212391 | -53.3482 | 12.6552 | HC3 | 21.5303 | -0.945018 | 0.356513 | -65.4099 | 24.7169 |

The fade outcome is **SOH percentage points lost per 100 valid cycles**. Full-span estimates support
temperature/current associations under the assumed additive model; short-window estimates are less conclusive.
The cycle-level model describes conditional log-SOH trajectories; this model describes between-battery estimated slopes,
so coefficients are not numerically interchangeable. Different run lengths, nonlinear early ageing and slope-estimation
uncertainty limit the comparison. HC3 does not propagate the first-stage slope uncertainty.

## C3.7: forecasting protocol, variants and model selection

For each held-out cell, `k=floor(0.60*n)` retained rows are history, origin is row k−1,
and the remaining rows form both E1 and E2 horizons. Forecast coordinates follow the locked group protocol.
E1 rolls observed history forward; E2 keeps the origin fixed and predicts only to the observed horizon endpoint.
No future SOH or measured operating condition is used as an E2 predictor. The retained future cycle coordinates
are the retrospective evaluation schedule specified by the group; they are not a predicted physical-time schedule.

The complete Tier A series, including quality-flagged rows, was checked for `ambient_C`, `I_level_A`, and
`discharge_cutoff_V`: all 36 battery/feature combinations (12 batteries × 3 features) contain exactly one finite value.
`regression_operating_conditions.csv` records the check. E2 constructs operating predictors from the origin values only.
Changing any condition after the origin, or introducing NaN/Inf, raises an explicit feature-availability error.
Time-varying conditions require an agreed known-at-origin schedule before this pipeline can support them.

| Candidate | Forecast predictors |
| --- | --- |
| research_loglinear | Train-centered cycle, 24°C indicator, centered current/cutoff, and cycle interactions with all three conditions |
| selected_loglinear | Hierarchy-selected subset of those terms, refitted inside each training fold; the final train fit retains all terms |
| quadratic_loglinear | Full research design plus centered cycle squared |
| anchored_loglinear | Research fitted log curve plus permitted observed SOH of this evaluation battery |
| anchored_quadratic | Quadratic fitted log curve plus permitted observed SOH of this evaluation battery |
| persistence | Previous retained observed SOH for E1; origin observed SOH for E2 |

All centers, coefficients, selection and non-anchored smearing factors use training data only.
Neither EIS, thermal measurements, charge duration nor future SOH enter the forecast design.

Prespecified candidates are full log-linear, hierarchy-selected log-linear, quadratic log-linear,
history-anchored full log-linear, and history-anchored quadratic log-linear. The quadratic variant adds one global cycle² term.
An anchored forecast is `SOH_reference × exp[f(cycle_target,conditions)−f(cycle_reference,conditions)]`.
It calibrates a battery-specific intercept using allowed history; the common smearing factor cancels in the ratio.
This ratio forecast is not guaranteed to equal a conditional arithmetic mean. No battery ID fitted effect is assigned to a new cell.
E1 uses the preceding retained row as reference; E2 uses only the fixed origin.

GroupKFold(k=4) on the eight train cells refits centering, selection and smearing in each fold.
Rank the average held-out-battery MAE, and freeze one winner per scenario before scoring test outcomes.
The E1 winner is **anchored_loglinear**; E2 winner is **anchored_quadratic**.
The common export label `regression` represents a **scenario-specific regression forecasting strategy** with
separate fitted variants and coefficient vectors. `regression_selected.json` stores the mapping, fitted formulas,
training centers and permitted anchors. Anchoring uses observed SOH from the test battery itself: only the preceding
retained observation for E1, and only the fixed origin for E2. It does not transfer a fitted battery intercept from training.
All scores, including losing candidates, are exported; the marginally lower test MAE of another candidate is not used to switch winners.
CV has only eight cells and one training cell at 24°C/4 A, making the ranking unstable and some CV conditions sparsely supported.

| scenario | model | MAE |
| --- | --- | --- |
| E1 | anchored_loglinear | 0.00947006 |
| E1 | anchored_quadratic | 0.00949914 |
| E1 | quadratic_loglinear | 0.0709579 |
| E1 | research_loglinear | 0.0696901 |
| E1 | selected_loglinear | 0.0911515 |
| E2 | anchored_loglinear | 0.0356176 |
| E2 | anchored_quadratic | 0.0353696 |
| E2 | quadratic_loglinear | 0.0709579 |
| E2 | research_loglinear | 0.0696901 |
| E2 | selected_loglinear | 0.0911515 |

### Equal-battery macro test metrics

| model | scenario | n | MAE | RMSE | MAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | E1 | 213 | 0.0141225 | 0.0201418 | 1.69765 | 0.0447127 |
| persistence | E2 | 213 | 0.0238145 | 0.0299687 | 2.9871 | -0.871726 |
| regression | E1 | 213 | 0.0137394 | 0.0202341 | 1.64797 | 0.0446581 |
| regression | E2 | 213 | 0.0335355 | 0.0420071 | 4.17526 | -2.75718 |
| research_loglinear | E1 | 213 | 0.0839132 | 0.0891713 | 10.5348 | -21.5502 |
| research_loglinear | E2 | 213 | 0.0839132 | 0.0891713 | 10.5348 | -21.5502 |

### Pooled-row micro test metrics

| model | scenario | n | MAE | RMSE | MAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | E1 | 213 | 0.0119048 | 0.0225923 | 1.4288 | 0.500084 |
| persistence | E2 | 213 | 0.0248857 | 0.034343 | 3.12707 | -0.15519 |
| regression | E1 | 213 | 0.0116455 | 0.02257 | 1.39527 | 0.501072 |
| regression | E2 | 213 | 0.0330108 | 0.0446499 | 4.12345 | -0.952615 |

### Per-battery test metrics

| model | scenario | battery_id | n | MAE | RMSE | MAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | E1 | B0007 | 67 | 0.00281157 | 0.00369149 | 0.3562 | 0.978299 |
| persistence | E1 | B0034 | 78 | 0.00850106 | 0.0134828 | 1.01564 | 0.425234 |
| persistence | E1 | B0047 | 28 | 0.0102344 | 0.0172037 | 1.29181 | 0.0751075 |
| persistence | E1 | B0055 | 40 | 0.0349428 | 0.0461893 | 4.12696 | -1.29979 |
| persistence | E2 | B0007 | 67 | 0.0451713 | 0.0515219 | 5.83987 | -3.22735 |
| persistence | E2 | B0034 | 78 | 0.0124125 | 0.0179084 | 1.48408 | -0.0140043 |
| persistence | E2 | B0047 | 28 | 0.0167467 | 0.0199398 | 2.16186 | -0.242473 |
| persistence | E2 | B0055 | 40 | 0.0209275 | 0.0305046 | 2.46261 | -0.00308113 |
| regression | E1 | B0007 | 67 | 0.00257595 | 0.00411958 | 0.326711 | 0.972974 |
| regression | E1 | B0034 | 78 | 0.00848306 | 0.0134791 | 1.01327 | 0.425551 |
| regression | E1 | B0047 | 28 | 0.00914341 | 0.0172873 | 1.15055 | 0.0661056 |
| regression | E1 | B0055 | 40 | 0.034755 | 0.0460506 | 4.10134 | -1.286 |
| regression | E2 | B0007 | 67 | 0.0559929 | 0.0636126 | 7.23028 | -5.44421 |
| regression | E2 | B0034 | 78 | 0.0145361 | 0.0213612 | 1.72348 | -0.442709 |
| regression | E2 | B0047 | 28 | 0.0332126 | 0.0408924 | 4.20468 | -4.22553 |
| regression | E2 | B0055 | 40 | 0.0304003 | 0.0421624 | 3.54258 | -0.916265 |

The anchored E1 variant greatly improves on the old uncalibrated cross-cell curve, but only slightly improves persistence MAE;
it does not improve every metric. The chosen E2 variant improves on the original curve but remains worse than frozen-origin persistence.
Direct log-linear E1 and E2 coincide because their predictors are the same. History anchoring changes the information used,
so gains must not be attributed solely to a better global regression specification.
Global test predictor ranges lie inside global training ranges. Nevertheless, sparse *joint* condition support and
cell-specific offsets/slopes make conditional extrapolation and heterogeneity plausible; the audit does not claim that scalar-range extrapolation caused failure.
No post-test candidate expansion or tuning was performed. No SVR, ARIMA or LSTM result is generated here.

### EOL/RUL

| model | battery_id | origin_cycle | horizon_end | observed_EOL_cycle | predicted_EOL_cycle | status | observed_RUL | predicted_RUL | signed_RUL_error | absolute_RUL_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | B0007 | 101 | 168 | 126 | — | prediction_no_crossing_in_horizon | 25 | — | — | — |
| persistence | B0034 | 118 | 196 | 186 | — | prediction_no_crossing_in_horizon | 68 | — | — | — |
| persistence | B0047 | 40 | 68 | 30 | 41 | already_at_or_before_origin | — | — | — | — |
| persistence | B0055 | 59 | 101 | 65 | — | prediction_no_crossing_in_horizon | 6 | — | — | — |
| regression | B0007 | 101 | 168 | 126 | 112 | both_cross_within_horizon | 25 | 11 | -14 | 14 |
| regression | B0034 | 118 | 196 | 186 | — | prediction_no_crossing_in_horizon | 68 | — | — | — |
| regression | B0047 | 40 | 68 | 30 | 41 | already_at_or_before_origin | — | — | — | — |
| regression | B0055 | 59 | 101 | 65 | 91 | both_cross_within_horizon | 6 | 32 | 26 | 26 |

The first observed `SOH ≤ 0.8` is absorbing for event timing even if SOH later recovers.
B0047 already crossed before the origin; its historical 27-cycle EOL error is invalid for this forecast task and is now blank.
No-crossing rows remain explicit, and RUL error is computed only for two within-horizon crossings.
No horizon extrapolation, interpolation of crossing cycles or silent removal of censored cells is performed.

## Outputs, integration and limitations

`reports/predictions/regression.csv` retains exactly `model,battery_id,cycle_valid,scenario,y_true,y_pred,y_lower,y_upper`.
It contains the CV-selected regression strategy, with `model=regression`; intervals are blank because no valid forecast interval is available.
Historical archives are optional. A valid old prediction CSV is scored separately in `regression_historical_metrics.csv`;
missing or invalid archives produce an explicit skip status and an empty historical table, never substitute predictions.
Current fit, CV, metrics, EOL results and the main prediction CSV are independent of archive availability.
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

## Verified methodological sources

- N. Duan, “Smearing Estimate: A Nonparametric Retransformation Method,” *JASA*, vol. 78, no. 383, pp. 605–610, 1983. [Publisher / DOI](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017).
- [scikit-learn GroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): non-overlapping validation groups.
- [statsmodels robust covariance documentation](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html): cluster correction, group degrees of freedom and t inference.
- [semopy model documentation](https://semopy.com/docs/model.html), [syntax](https://semopy.com/syntax.html) and [fit-statistic implementation](https://semopy.com/docs/stats.html): covariance specification, estimation and parameter-count degrees of freedom. Checked against installed semopy source.

These sources document the methods. They do not establish battery-specific causal mechanisms; the team's battery literature review still needs substantive primary studies.

## Final stabilization validation (2026-11-10)

All **28 automated tests passed** (0 failures/errors); six Python files and all notebook code cells compiled.
Both complete notebooks ran with fresh kernels from the repository root and from `notebooks/`
(13 regression and 4 SEM code cells, zero error outputs); numerical CSV/JSON artifacts matched exactly between working directories.
An isolated repository copy without either historical archive also executed both complete notebooks and generated all three reports.
All **57 current-result CSVs**, including the main prediction CSV, matched the archive-present run byte for byte.
Historical-only tables were explicitly empty with skip messages. The report generator also succeeded without historical inputs.

The existing isolated `.venv` passed all 98 installed dependency constraints checked using `importlib.metadata` and `packaging`;
it has no `pip` module, so `pip check` was unavailable. No dependency or system environment was changed.
SHA256 checks confirmed 35 protected files unchanged, including original/processed data, shared pipeline, `Rec.md`,
the preexisting test notebook and real historical archives. `git diff --check` passed.
Detailed evidence: [final validation record](../reports/audit/regression_sem_final_validation.json) and [test log](../reports/audit/regression_sem_final_tests.log).

Compared with the preserved pre-pass files, the maximum prediction change was 6.85e-14 SOH and the maximum
bootstrap effect-CI endpoint change was 6.13e-7. A control run using the unchanged pre-pass source reproduced the new
prediction and SEM CSVs exactly: these differences arise on numerical reexecution, not from model/specification changes.
The unscaled legacy SEM sensitivity changed substantially (objective about 0.590259 in the preserved table versus
6.242571 now), also exactly reproduced by unchanged-source control. Its scale-sensitive optimizer success flag does not
establish an adequate solution; the internally scaled primary SEM remains the reported analysis.
Current regression metric/EOL tables omit historical rows; historical prediction metrics are now separate.
The preserved `sem_historical_raw_units_coefficients.csv` is a legacy artifact, not an output consumed by the current pipeline.

Review remains necessary for the scenario-specific anchoring strategy before group comparison, alignment with course slide p.39,
the theoretical SEM paths, incomplete condition crossing, retrospective cleaning, absence of calibrated prediction intervals,
E2 performance below persistence, and the limited information from 22 battery blocks. No causal mediation or good primary SEM fit is claimed.
