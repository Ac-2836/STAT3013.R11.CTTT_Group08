"""LSTM forecasting the SOH of the next cycle (Khải Hưng; WBS C6.7-C6.9, docs/Go.md GĐ5d).

The target cycle t is predicted from a window of the WINDOW cycles t-9..t. Each step of the window
holds what is known just before that cycle's discharge, the same information set as the SVR features
(src/svr_model.py on branch minh-vu):
    soh_prev, T_max_prev_C                      result of the previous discharge
    chg_cc_time_s, Re_ohm, Rct_ohm, log_rest_h  measured before this discharge
    is_4C, I_level_A, cutoff_V                  fixed operating conditions
The network learns dsoh = soh_t - soh_{t-1}; soh_prev + dsoh is the SOH forecast (docs/Rec.md 6.1 #8).

Training (C6.8): LSTM(1 layer, 32 units) -> dropout 0.2 -> Linear(1), Adam, MSE on the scaled ΔSOH.
GroupKFold by battery (k = 4) on the training batteries gives early stopping (patience 20), the learning
curves and the validation score; the final model is refit on all training batteries for the median best
epoch of the folds. Everything is repeated for 5 seeds (docs/Rec.md 6.1 #7).

Run from the repository root to build the windows and run the self-checks:
    python src/lstm_model.py
"""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from protocol import KEYS, PRED_COLS, load_split

HELD = ["chg_cc_time_s", "Re_ohm", "Rct_ohm", "log_rest_h"]   # measured, unknown after the E2 origin
HIDDEN, DROPOUT, LR, BATCH, MAX_EPOCHS, PATIENCE = 32, 0.2, 1e-3, 32, 300, 20
SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 4
WINDOW = 10
STEP_FEATURES = ["soh_prev", "T_max_prev_C", "chg_cc_time_s", "Re_ohm", "Rct_ohm", "log_rest_h",
                 "is_4C", "I_level_A", "cutoff_V"]
FORBIDDEN = {"soh", "capacity_Ah", "capacity_to_2p7V_Ah", "capacity_int_Ah", "energy_Wh", "duration_s",
             "t_to_cutoff_s", "T_max_C", "V_min", "C0_Ah", "soh_segment"}   # same cycle, docs/Go.md 4.6


def step_features(df):
    """One row of step features per cycle; lags are taken within each battery."""
    g = df.groupby("battery_id")
    return pd.DataFrame({
        "soh_prev": g.soh.shift(1),
        "T_max_prev_C": g.T_max_C.shift(1),
        "chg_cc_time_s": df.chg_cc_time_s,
        "Re_ohm": df.Re_ohm,
        "Rct_ohm": df.Rct_ohm,
        "log_rest_h": np.log1p(df.rest_before_h),
        "is_4C": (df.ambient_C < 10).astype(float),     # tier A has only 4°C and 24°C
        "I_level_A": df.I_level_A,
        "cutoff_V": df.discharge_cutoff_V,
    }, index=df.index)[STEP_FEATURES]


def make_windows(df):
    """Return (X, dsoh, soh_prev, idx).

    X has shape (n, WINDOW, len(STEP_FEATURES)); idx is the df index of each window's target row.
    df must be sorted by battery_id and cycle_valid with a 0..n-1 index, as load_split() returns it.
    Windows are built inside each battery, so none crosses two batteries; windows holding a missing
    value (Re/Rct before the first EIS of B0005-B0007) are dropped, not filled (docs/Rec.md 6.1 #6).
    """
    F = step_features(df)
    X, idx = [], []
    for rows in df.groupby("battery_id").groups.values():
        f = F.loc[rows].to_numpy()
        for end in range(WINDOW, len(rows)):            # row 0 has no soh_prev
            w = f[end - WINDOW + 1:end + 1]
            if not np.isnan(w).any():
                X.append(w)
                idx.append(rows[end])
    idx = np.array(idx)
    soh_prev = F.soh_prev.loc[idx].to_numpy()
    return np.stack(X).astype(np.float32), df.soh.loc[idx].to_numpy() - soh_prev, soh_prev, idx


def check_windows(df, X, dsoh, soh_prev, idx):
    """Window rules of WBS C6.7: one battery per window, consecutive cycles, no same-cycle leakage."""
    assert not FORBIDDEN & set(STEP_FEATURES)
    assert list(df.index) == list(range(len(df))), "df needs the 0..n-1 index of load_split()"
    start = idx - WINDOW + 1
    assert (df.battery_id.to_numpy()[start] == df.battery_id.to_numpy()[idx]).all(), "window crosses batteries"
    assert (df.groupby("battery_id").cumcount().to_numpy()[start] >= 1).all(), "window starts before soh_prev exists"
    assert not np.isnan(X).any()
    assert np.allclose(X[:, -1, 0], soh_prev), "last step must carry soh_{t-1}, not soh_t"
    assert np.allclose(soh_prev + dsoh, df.soh.loc[idx]), "target must rebuild soh_t"


class LSTMReg(nn.Module):
    """docs/Go.md GĐ5d: the last hidden state of the window -> dropout -> one ΔSOH value."""

    def __init__(self, n_feat, h=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, h, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(h, 1)

    def forward(self, x):                  # x: (batch, WINDOW, n_feat)
        _, (hT, _) = self.lstm(x)
        return self.out(self.drop(hT[-1])).squeeze(-1)


def fit_scale(X, y):
    """Mean/SD of every step feature and SD of ΔSOH, from training windows only."""
    flat = X.reshape(-1, X.shape[-1])
    sd = flat.std(0)
    return flat.mean(0), np.where(sd > 0, sd, 1), y.std()   # a fold may hold one condition only


def scale_X(X, scale):
    mu, sd, _ = scale
    return ((X - mu) / sd).astype(np.float32)


def mse(model, X, y):
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(torch.from_numpy(X)), torch.from_numpy(y)).item()


def train_one(X, y, seed, X_val=None, y_val=None, epochs=MAX_EPOCHS):
    """Train on scaled windows and targets. With a validation set: stop after PATIENCE epochs without
    improvement and restore the best weights. Returns (model, history of train/val MSE, best epoch)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = LSTMReg(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    y_val = None if y_val is None else y_val.astype(np.float32)
    hist, best, best_state, best_epoch, wait = [], np.inf, None, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(Xt), generator=gen).split(BATCH):
            opt.zero_grad()
            nn.functional.mse_loss(model(Xt[b]), yt[b]).backward()
            opt.step()
        va = np.nan if X_val is None else mse(model, X_val, y_val)
        hist.append((epoch, mse(model, X, y.astype(np.float32)), va))
        if X_val is None:
            continue
        if va < best:
            best, best_state, best_epoch, wait = va, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        assert np.isclose(mse(model, X_val, y_val), best), "best weights were not restored"
    return model, pd.DataFrame(hist, columns=["epoch", "train_mse", "val_mse"]), best_epoch


def predict_dsoh(model, X, scale):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(scale_X(X, scale))).numpy().astype(float) * scale[2]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cross_validate(train, seeds=SEEDS):
    """GroupKFold by battery on the training batteries, for every seed.

    Returns (folds, curves): folds has one row per seed and fold with the best epoch and the one-step
    SOH RMSE of the LSTM and of naive on the validation batteries; curves holds every learning curve.
    """
    X, dsoh, soh_prev, idx = make_windows(train)
    groups = train.battery_id.to_numpy()[idx]
    folds, curves = [], []
    for seed in seeds:
        for k, (tr, va) in enumerate(GroupKFold(N_SPLITS).split(X, dsoh, groups)):
            scale = fit_scale(X[tr], dsoh[tr])
            model, hist, best_epoch = train_one(scale_X(X[tr], scale), dsoh[tr] / scale[2], seed,
                                                scale_X(X[va], scale), dsoh[va] / scale[2])
            soh_true = soh_prev[va] + dsoh[va]
            folds.append({"seed": seed, "fold": k, "val_batteries": ", ".join(sorted(set(groups[va]))),
                          "n_val": len(va), "best_epoch": best_epoch,
                          "rmse_lstm": rmse(soh_prev[va] + predict_dsoh(model, X[va], scale), soh_true),
                          "rmse_naive": rmse(soh_prev[va], soh_true)})
            curves.append(hist.assign(seed=seed, fold=k))
    return pd.DataFrame(folds), pd.concat(curves, ignore_index=True)


def fit_final(train, seed, epochs):
    """Refit on all training batteries for a fixed number of epochs; returns (model, scale)."""
    X, dsoh, _, _ = make_windows(train)
    scale = fit_scale(X, dsoh)
    model, _, _ = train_one(scale_X(X, scale), dsoh / scale[2], seed, epochs=epochs)
    return model, scale


def forecast_e1(model, scale, test):
    """One step ahead for every is_eval row: the window holds true values up to t-1."""
    X, _, soh_prev, idx = make_windows(test)
    ev = test.is_eval.to_numpy()[idx]
    return pd.Series(soh_prev[ev] + predict_dsoh(model, X[ev], scale), index=idx[ev])


def forecast_e2(model, scale, battery):
    """Recursive forecast of one test battery from its origin (first is_eval row), using only earlier rows.

    soh_prev is fed back from the forecast; HELD features keep their last value before the origin and
    T_max_prev_C keeps the last one known at the origin (docs/Rec.md 6.1 #5, as in the SVR).
    """
    o = int(np.argmax(battery.is_eval.to_numpy()))
    F = step_features(battery).to_numpy(dtype=float, copy=True)
    col = {c: i for i, c in enumerate(STEP_FEATURES)}
    for c in HELD:
        F[o:, col[c]] = F[o - 1, col[c]]
    F[o + 1:, col["T_max_prev_C"]] = F[o, col["T_max_prev_C"]]
    out = []
    for i in range(o, len(F)):
        w = F[i - WINDOW + 1:i + 1][None]
        assert i - WINDOW + 1 >= 1 and not np.isnan(w).any()
        soh = F[i, col["soh_prev"]] + predict_dsoh(model, w, scale)[0]
        out.append(soh)
        if i + 1 < len(F):
            F[i + 1, col["soh_prev"]] = soh
    return pd.Series(out, index=battery.index[o:])


def check_e2_no_peeking(model, scale, battery):
    """Scrambling every true value after the origin must not change the E2 forecast."""
    after = battery.is_eval.to_numpy()
    cols = ["soh", "T_max_C", "chg_cc_time_s", "Re_ohm", "Rct_ohm", "rest_before_h"]
    scrambled = battery.copy()
    scrambled.loc[after, cols] = scrambled.loc[after, cols].to_numpy()[::-1] * 0.9
    assert np.allclose(forecast_e2(model, scale, battery), forecast_e2(model, scale, scrambled)), \
        "E2 forecast uses values after the origin"


def to_pred(test, preds, model):
    """{scenario: Series indexed like test} -> rows in the shared prediction format."""
    parts = []
    for scenario, s in preds.items():
        rows = test.loc[s.index]
        parts.append(rows[KEYS].assign(model=model, scenario=scenario, y_true=rows.soh, y_pred=s.to_numpy(),
                                       y_lower=np.nan, y_upper=np.nan)[PRED_COLS])
    return pd.concat(parts, ignore_index=True)


def run_test(train, test, final_epochs):
    """Refit one model per seed ({seed: epochs}) and forecast E1 and E2 on the test batteries.

    Returns (per_seed, lstm): per-seed predictions named lstm_seed<k>, and the seed-mean predictions
    named lstm that go to reports/predictions/lstm.csv. The LSTM gives no prediction interval.
    """
    per_seed = []
    for seed, epochs in final_epochs.items():
        model, scale = fit_final(train, seed, epochs)
        e1 = forecast_e1(model, scale, test)
        e2 = pd.concat([forecast_e2(model, scale, b) for _, b in test.groupby("battery_id")])
        per_seed.append(to_pred(test, {"E1": e1, "E2": e2}, f"lstm_seed{seed}"))
    per_seed = pd.concat(per_seed, ignore_index=True)
    lstm = (per_seed.groupby(["scenario"] + KEYS, as_index=False)
            .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
            .assign(model="lstm", y_lower=np.nan, y_upper=np.nan)[PRED_COLS])
    return per_seed, lstm


def main():
    train, test = load_split()
    model, scale = fit_final(train, SEEDS[0], 20)
    for _, b in test.groupby("battery_id"):
        check_e2_no_peeking(model, scale, b)
    for name, df in (("train", train), ("test", test)):
        X, dsoh, soh_prev, idx = make_windows(df)
        check_windows(df, X, dsoh, soh_prev, idx)
        per = df.loc[idx].battery_id.value_counts().sort_index().to_dict()
        print(f"{name}: X {X.shape}, windows per battery {per}")
        if name == "train":
            assert len(X) == 867 - 2 * 18          # 18 windows of B0005 and B0006 touch the missing Re/Rct
        else:
            assert set(test.index[test.is_eval]) <= set(idx), "every evaluated row needs a window"


if __name__ == "__main__":
    main()
