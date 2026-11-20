# LSTM methods and results — Khai Hung

Draft text for Full Report §3.6 (methodology) and §5.6 (results), WBS C6.10. Every number comes from the
executed outputs of `notebooks/40_dl.ipynb` (code in `src/lstm_model.py`, shared protocol in `src/protocol.py`).
Decisions referenced as "Rec.md 6.1" are recorded in `docs/Rec.md` section 6.1.

---

## 3.6 Long Short-Term Memory network (LSTM)

### 3.6.1 Purpose

The LSTM answers RQ5: *does a deep sequence model learn more from the feature history than a machine-learning
model on a small dataset?* It forecasts the SOH of the next cycle from a window of recent cycles and is
evaluated on the same rows, scenarios and metrics as the other three models (§3.2).

### 3.6.2 Recurrent networks and the LSTM cell

A recurrent neural network reads a sequence one step at a time and keeps a hidden state that summarises the
steps seen so far:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b).$$

Training back-propagates the error through all steps (back-propagation through time). Because the gradient is
multiplied by a similar factor at every step, it tends to vanish over long sequences, so a plain RNN learns
little from early steps.

The LSTM [Hochreiter & Schmidhuber, 1997] adds a cell state $c_t$ controlled by three sigmoid gates:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{forget gate: how much of } c_{t-1} \text{ to keep}\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{input gate: how much new content to write}\\
g_t &= \tanh(W_g [h_{t-1}, x_t] + b_g) && \text{candidate content}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t && \text{cell update}\\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{output gate}\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

The cell is updated additively, so along the cell path the gradient is scaled only by $f_t$; when the network
keeps $f_t \approx 1$ the gradient passes through many steps without vanishing. In the battery setting this
allows the network, in principle, to keep the long-term fade trend in $c_t$ while discounting a temporary
capacity recovery after a long rest through the input gate.

### 3.6.3 Input windows

The target cycle $t$ is predicted from a window of the $L = 10$ cycles $t-9, \dots, t$. Each step holds only
what is known just before that cycle's discharge — the same information set as the SVR features (§3.5):

| Step feature | Meaning |
|---|---|
| `soh_prev`, `T_max_prev_C` | SOH and maximum cell temperature of the previous discharge |
| `chg_cc_time_s`, `Re_ohm`, `Rct_ohm`, `log_rest_h` | constant-current charge time of the preceding charge, last measured impedance, $\ln(1 + \text{rest hours})$ |
| `is_4C`, `I_level_A`, `cutoff_V` | operating conditions, fixed per battery |

Rules (checked by assertions in the code):

- windows are built inside one battery and never cross two batteries;
- no feature of the target cycle's own discharge is used (capacity, energy, duration, `T_max_C`, `V_min`);
- windows containing missing impedance values (the 19 cycles before the first EIS measurement of
  B0005–B0007) are dropped, not imputed, because filling them would use later measurements (Rec.md 6.1 #6).

This gives 831 training windows (867 minus 18 for each of B0005 and B0006) and 470 test windows that cover
all 213 evaluated rows.

### 3.6.4 Target

The network predicts the change $\Delta\text{SOH}_t = \text{SOH}_t - \text{SOH}_{t-1}$ and the forecast is
$\widehat{\text{SOH}}_t = \text{SOH}_{t-1} + \widehat{\Delta\text{SOH}}_t$ (Rec.md 6.1 #8). Predicting the level
directly lets a network score a high $R^2$ by copying the previous value, which is exactly the naive forecast;
predicting the change forces it to learn what naive does not know. On the training windows ΔSOH has mean
−0.0019 and SD 0.0144 per cycle, with jumps up to +0.125 after long rests, so the signal is small relative to
the noise.

### 3.6.5 Architecture

`LSTM(9 → 32, one layer)` → `Dropout(0.2)` on the last hidden state → `Linear(32 → 1)`, 5,537 trainable
parameters. The model is deliberately small: the parameter count already exceeds the 831 training windows,
and the course material notes that deep learning needs large datasets.

### 3.6.6 Training and model selection

- Inputs are standardised per feature and the target is divided by its SD; both scalings are fitted on the
  training part of each fold only.
- Adam (learning rate 0.001), MSE loss, mini-batches of 32, at most 300 epochs.
- Validation: `GroupKFold` by battery with $k = 4$ on the 8 training batteries (two batteries held out per
  fold), the same scheme as the SVR (Rec.md 6.1 #7). Early stopping with patience 20 restores the weights of
  the best validation epoch.
- The final model of each seed is refitted on all 8 training batteries for the median best epoch of its four
  folds.
- Everything is repeated for 5 seeds (42–46) and reported as mean ± SD. The file
  `reports/predictions/lstm.csv` holds the mean forecast of the 5 seeds.

Learning rate and batch size were fixed, not tuned.

### 3.6.7 Forecasting scenarios

- **E1, one step ahead:** each of the 213 evaluated rows is predicted from a window of true values up to $t-1$.
- **E2, recursive:** from the 60% origin of each test battery, the forecast SOH is fed back as `soh_prev` of
  the next step. The measured features (`chg_cc_time_s`, `Re_ohm`, `Rct_ohm`, `log_rest_h`, `T_max_prev_C`)
  keep their last value known at the origin (Rec.md 6.1 #5). A test scrambles every true value after the
  origin and confirms that the E2 forecast does not change.

The LSTM gives no prediction interval.

---

## 5.6 LSTM results

### 5.6.1 Validation on the training batteries

Pooled out-of-fold one-step SOH RMSE over the four validation folds:

| | RMSE (mean ± SD over 5 seeds) |
|---|---|
| LSTM | 0.0121 ± 0.0001 |
| Naive | 0.0145 |

The LSTM is about 17% below naive and beats it in 19 of the 20 fold–seed runs; the exception (fold B0033 +
B0045, seed 45) is 0.0122 against 0.0121. The best epoch varies widely between folds (6 to 116), so the refit
uses 16 to 33 epochs depending on the seed. The fold holding B0018 and B0056 has a much higher validation loss
than the others (naive RMSE 0.0202), reflecting noisier batteries rather than over-fitting (Figure:
`reports/figures/lstm_learning_curve.png`).

### 5.6.2 Test batteries

Pooled over B0007, B0034, B0047 and B0055 (213 rows per scenario):

| Model | Scenario | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| LSTM (seed mean) | E1 | **0.0098** | **0.0190** | **1.18** | **0.647** |
| Naive | E1 | 0.0119 | 0.0226 | 1.43 | 0.500 |
| LSTM (seed mean) | E2 | 0.0303 | 0.0383 | 3.74 | −0.436 |
| Naive | E2 | **0.0249** | **0.0343** | **3.13** | **−0.155** |

Single seeds: E1 MAE 0.0100 ± 0.0002, RMSE 0.0191 ± 0.0003, R² 0.641 ± 0.011; E2 MAE 0.0393 ± 0.0125,
RMSE 0.0523 ± 0.0172, R² −1.91 ± 2.02.

RMSE per test battery:

| Battery (condition) | LSTM E1 | Naive E1 | LSTM E2 | Naive E2 |
|---|---|---|---|---|
| B0007 (24°C, 2 A) | 0.0025 | 0.0037 | 0.0361 | 0.0515 |
| B0034 (24°C, 4 A) | 0.0085 | 0.0135 | 0.0381 | 0.0179 |
| B0047 (4°C, 1 A) | 0.0089 | 0.0172 | 0.0489 | 0.0199 |
| B0055 (4°C, 2 A) | 0.0414 | 0.0462 | 0.0335 | 0.0305 |

End of life (first cycle with SOH < 0.8), E2:

| Battery | True EOL cycle | LSTM EOL cycle | RUL error (cycles) |
|---|---|---|---|
| B0007 | 126 | 118 | −8 (early) |
| B0034 | 186 | 150 | −36 (early) |
| B0055 | 65 | not reached | — |

B0047 is excluded from RUL because its SOH is already below 0.8 before the origin (Rec.md 6.1 #3). Naive never
predicts EOL, so it gives no RUL.

### 5.6.3 Discussion

- **One step ahead the LSTM helps.** It beats naive on all four test batteries, and single seeds agree (RMSE SD 0.0003).
  The forecast plot (`reports/figures/lstm_forecasts.png`) shows why the gain is modest: the E1 curve mostly
  follows the previous true value with a small downward correction, so sudden capacity recoveries are
  reproduced one cycle late. The largest relative gains are at 4°C/1 A (B0047) and 24°C/4 A (B0034).
- **Over many steps it does not.** In E2 the LSTM extrapolates a smooth, almost linear decline. This is right
  for the steadily fading B0007, where it beats naive and predicts EOL 8 cycles early, but on B0034 and B0047
  the forecast falls faster than the true SOH, which keeps recovering after rests. Once the measured features
  are frozen at the origin, the network has no information about future recoveries.
- **Averaging seeds matters for recursive forecasts.** Single seeds drift in different directions in E2
  (RMSE SD 0.0172); the seed mean (0.0383) is much better than the average seed (0.0523).
- **Practical reading.** An early EOL warning is the safe side for warranty and fleet decisions, but a warning
  36 cycles early (B0034) would retire batteries too soon. The LSTM is therefore useful as a short-term
  monitor, not as a stand-alone RUL estimator.
- **Answer to RQ5 (LSTM side).** With about 830 training windows the LSTM learns a useful one-step correction
  but not a reliable long-horizon trajectory. The comparison with the SVR, regression and ARIMA is in §5.8.

### 5.6.4 Limitations

- Small data: 8 training batteries, 831 windows, 5,537 parameters; early stopping, not data volume, limits
  over-fitting.
- Learning rate, batch size, window length and hidden size were fixed by the plan, not tuned.
- The benchmark is retrospective: tier membership and the centred Hampel outlier flags use each battery's
  full record, so the cleaned test series is slightly smoother than what an online system would see.
- Test results were not used for any model choice; tuning on them now would leak the test set.

### Reference

S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8,
pp. 1735–1780, 1997.
