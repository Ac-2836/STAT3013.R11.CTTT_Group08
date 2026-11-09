"""LSTM forecasting the SOH of the next cycle (Khải Hưng; WBS C6.7-C6.9, docs/Go.md GĐ5d).

The target cycle t is predicted from a window of the WINDOW cycles t-9..t. Each step of the window
holds what is known just before that cycle's discharge, the same information set as the SVR features
(src/svr_model.py on branch minh-vu):
    soh_prev, T_max_prev_C                      result of the previous discharge
    chg_cc_time_s, Re_ohm, Rct_ohm, log_rest_h  measured before this discharge
    is_4C, I_level_A, cutoff_V                  fixed operating conditions
The network learns dsoh = soh_t - soh_{t-1}; soh_prev + dsoh is the SOH forecast (docs/Rec.md 6.1 #8).

Run from the repository root to build the windows and run the self-checks:
    python src/lstm_model.py
"""
import numpy as np
import pandas as pd

from protocol import load_split

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


def main():
    train, test = load_split()
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
