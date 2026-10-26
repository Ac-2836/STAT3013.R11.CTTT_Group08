"""Clean the joined discharge table, compute C0/SOH, flag outliers and assign tiers (preprocessing steps 6-10).

Reads   data/processed/cycles.csv           (from build_dataset.py)
Writes  data/processed/cycles_clean.csv     one row per valid discharge (2,658)
        data/processed/cycles_dropped.csv   the 136 dropped discharges with their drop_reason
        data/processed/battery_summary.csv  one row per cell (34) with its tier

All rules and thresholds follow docs/Rec.md sections 2 and 2.1. Run from the repository root:
    python src/clean_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

EXCLUDED_GROUP = "G8"
CUTOFF_MARGIN_V = 0.1
CAP_RANGE_AH = (0.05, 2.1)
R_MAX_OHM = 0.5
HAMPEL_WINDOW, HAMPEL_K, HAMPEL_MIN_AH = 7, 3, 0.05
DROP_WINDOW, DROP_RATIO = 10, 0.75
BREAK_IN_CYCLES, BREAK_IN_RATIO = 10, 0.9
LONG_REST_H = 1
TIER_MIN_CYCLES, TIER_MAX_SOH_MIN, TIER_MAX_OUTLIER_SHARE = 60, 0.85, 0.10


def drop_invalid(cycles):
    """Step 6: the first matching rule, in this fixed order, is the drop reason."""
    reason = np.select(
        [cycles.group == EXCLUDED_GROUP,
         cycles.chg_op_index.isna(),
         cycles.V_min > cycles.discharge_cutoff_V + CUTOFF_MARGIN_V,
         ~cycles.capacity_int_Ah.between(*CAP_RANGE_AH)],
        ["excluded_battery", "no_preceding_charge", "cutoff_not_reached", "capacity_out_of_range"],
        default="",
    )
    dropped = cycles[reason != ""].assign(drop_reason=reason[reason != ""])
    return cycles[reason == ""].copy(), dropped


def filter_impedance(df):
    """Step 7a: keep Re/Rct only inside (0, 0.5) ohm, then carry the last valid value forward within each cell."""
    ok = (df.Re_ohm.between(0, R_MAX_OHM, inclusive="neither")
          & df.Rct_ohm.between(0, R_MAX_OHM, inclusive="neither"))
    df[["Re_ohm", "Rct_ohm"]] = df[["Re_ohm", "Rct_ohm"]].where(ok)
    df[["Re_ohm", "Rct_ohm"]] = df.groupby("battery_id")[["Re_ohm", "Rct_ohm"]].ffill()  # never bfill: no future values


def add_segments(df):
    """Step 7b/c: a new segment starts whenever (ambient_C, I_level_A) changes; cycle_valid counts valid rows per cell."""
    g = df.groupby("battery_id")
    changed = (df.ambient_C != g.ambient_C.shift()) | (df.I_level_A != g.I_level_A.shift())
    df["segment"] = changed.astype(int).groupby(df.battery_id).cumsum()
    df["cycle_valid"] = g.cumcount() + 1


def c0(s):
    """Max of the trailing 3-cycle rolling median over the first 10 valid cycles of a segment."""
    r = s.head(10).rolling(3, min_periods=1).median()
    return r.iloc[2:].max() if len(r) >= 3 else r.max()   # positions 0-1 are not true 3-point medians


def add_soh(df):
    """Step 8: C0 per (cell, segment); soh only for single-segment cells."""
    df["C0_Ah"] = df.groupby(["battery_id", "segment"]).capacity_int_Ah.transform(c0)
    df["soh_segment"] = df.capacity_int_Ah / df.C0_Ah
    single = df.groupby("battery_id").segment.transform("max") == 1
    df["soh"] = df.soh_segment.where(single)


def hampel(x):
    med = x.rolling(HAMPEL_WINDOW, center=True, min_periods=3).median()
    dev = (x - med).abs()
    mad = dev.rolling(HAMPEL_WINDOW, center=True, min_periods=3).median() * 1.4826
    return dev > np.maximum(HAMPEL_K * mad, HAMPEL_MIN_AH)


def sudden_drop(x):
    prev_max = x.shift().rolling(DROP_WINDOW, min_periods=3).max()
    return x < DROP_RATIO * prev_max


def add_flags(df):
    """Step 9: flags are kept as columns, rows are not deleted."""
    cap = df.groupby(["battery_id", "segment"]).capacity_int_Ah
    df["cap_outlier"] = cap.transform(hampel).astype(bool) | cap.transform(sudden_drop).astype(bool)
    df["break_in"] = (df.cycle_valid <= BREAK_IN_CYCLES) & (df.capacity_int_Ah < BREAK_IN_RATIO * df.C0_Ah)
    df["long_rest"] = df.rest_before_h > LONG_REST_H


def summarize(cycles, clean):
    """Step 10: one row per cell, tiers decided in the order X -> C -> N -> A -> B."""
    s = cycles.groupby("battery_id").agg(group=("group", "first"), profile=("profile", "first"),
                                         discharge_cutoff_V=("discharge_cutoff_V", "first"),
                                         n_discharges_raw=("op_index", "size"))
    s["n_dropped"] = s.n_discharges_raw - clean.groupby("battery_id").size().reindex(s.index, fill_value=0)

    nb = clean[~clean.break_in]
    first = clean.groupby("battery_id").first()
    s["n_segments"] = clean.groupby("battery_id").segment.max()
    single = s.n_segments == 1
    s["ambient_C"] = first.ambient_C.where(single)
    s["I_level_A"] = first.I_level_A.where(single)
    s["n_valid"] = nb.groupby("battery_id").size()
    s["n_cap_outlier"] = nb.groupby("battery_id").cap_outlier.sum()
    s["outlier_share"] = s.n_cap_outlier / s.n_valid
    s["n_break_in"] = clean.groupby("battery_id").break_in.sum()
    s["C0_Ah"] = first.C0_Ah.where(single)
    s["soh_min"] = nb[~nb.cap_outlier].groupby("battery_id").soh.min()
    s["soh_last"] = nb.groupby("battery_id").soh.last()   # keeps cap_outlier rows (battery_fade.csv does not)
    s["tier"] = np.select(
        [s.group == EXCLUDED_GROUP,
         s.n_segments > 1,
         s.outlier_share > TIER_MAX_OUTLIER_SHARE,
         (s.n_valid >= TIER_MIN_CYCLES) & (s.soh_min <= TIER_MAX_SOH_MIN)],
        ["X", "C", "N", "A"],
        default="B",
    )
    counts = ["n_segments", "n_valid", "n_cap_outlier", "n_break_in"]
    s[counts] = s[counts].astype("Int64")   # integer counts, empty (not 1.0-style floats) for tier X cells
    cols = ["group", "profile", "discharge_cutoff_V", "ambient_C", "I_level_A",
            "n_discharges_raw", "n_dropped", "n_segments", "n_valid", "n_cap_outlier", "outlier_share",
            "n_break_in", "C0_Ah", "soh_min", "soh_last", "tier"]
    return s[cols].reset_index()


def check(clean, dropped, summary):
    assert dropped.drop_reason.value_counts().to_dict() == {
        "excluded_battery": 100, "no_preceding_charge": 19, "cutoff_not_reached": 16, "capacity_out_of_range": 1}
    assert len(clean) == 2658
    assert (clean.cap_outlier.sum(), clean.break_in.sum(), clean.long_rest.sum()) == (57, 17, 167)
    assert clean.Re_ohm.isna().sum() == 57
    assert clean.soh.notna().sum() == 2121
    assert summary.tier.value_counts().to_dict() == {"A": 12, "B": 10, "C": 7, "N": 1, "X": 4}
    assert clean.groupby("tier").size().to_dict() == {"A": 1503, "B": 517, "C": 537, "N": 101}
    c0 = summary.set_index("battery_id").loc[["B0005", "B0033", "B0054"], "C0_Ah"].round(3).tolist()
    assert c0 == [1.852, 1.797, 1.483], c0


def main():
    cycles = pd.read_csv(DATA / "cycles.csv", parse_dates=["start_time", "chg_end_time"])
    clean, dropped = drop_invalid(cycles)
    clean = clean.sort_values(["battery_id", "op_index"]).reset_index(drop=True)   # every step below relies on this order
    filter_impedance(clean)
    add_segments(clean)
    add_soh(clean)
    add_flags(clean)
    summary = summarize(cycles, clean)
    clean["tier"] = clean.battery_id.map(summary.set_index("battery_id").tier)

    cols = ["battery_id", "group", "op_index", "cycle", "cycle_valid", "segment", "start_time",
            "ambient_C", "I_level_A", "profile", "discharge_cutoff_V",
            "capacity_Ah", "capacity_to_2p7V_Ah", "capacity_int_Ah", "C0_Ah", "soh_segment", "soh",
            "I_load_mean_A", "V_min", "duration_s", "t_to_cutoff_s", "energy_Wh", "T_start_C", "T_max_C", "T_mean_C",
            "chg_op_index", "chg_duration_s", "chg_end_time", "chg_charge_Ah", "chg_cc_time_s", "chg_T_max_C",
            "rest_before_h", "n_chg_since_prev_dis",
            "eis_op_index", "Re_ohm", "Rct_ohm",
            "cap_outlier", "break_in", "long_rest", "tier"]
    clean = clean[cols]
    check(clean, dropped, summary)

    clean.to_csv(DATA / "cycles_clean.csv", index=False)
    dropped.to_csv(DATA / "cycles_dropped.csv", index=False)
    summary.to_csv(DATA / "battery_summary.csv", index=False)
    print(f"cycles_clean {len(clean)}, cycles_dropped {len(dropped)}, battery_summary {len(summary)} rows "
          f"-> {DATA.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
