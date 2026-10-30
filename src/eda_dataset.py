"""Cell-level fade table for inferential statistics (preprocessing step 11).

Reads   data/processed/cycles_clean.csv   (from clean_dataset.py)
Writes  data/processed/battery_fade.csv   one row per constant-condition cell (tiers A, B, N: 23 cells)

The cell, not the cycle, is the unit of observation for the tests, because cycles of one
cell are strongly autocorrelated (docs/Introduction_of_dataset.md F9).
Descriptive statistics, ADF tests and figures will be added to this script later.

Run from the repository root:
    python src/eda_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

FADE_TIERS = ["A", "B", "N"]
WINDOW25 = 25
EOL_SOH = 0.8


def fade_per_100(g):
    """SOH percentage points lost per 100 cycles, from a linear fit of soh on cycle_valid."""
    slope = np.polyfit(g.cycle_valid, g.soh, 1)[0]
    return -slope * 100 * 100


def fade_row(g):
    g25 = g[g.cycle_valid <= WINDOW25]
    below = g.loc[g.soh < EOL_SOH, "cycle_valid"]
    first = g.iloc[0]
    return pd.Series({
        "group": first.group, "tier": first.tier, "ambient_C": first.ambient_C, "I_level_A": first.I_level_A,
        "profile": first.profile, "discharge_cutoff_V": first.discharge_cutoff_V,
        "n_used": len(g),
        "fade_pct_per_100cyc": fade_per_100(g),
        "n_used25": len(g25),               # can be < 25: flagged rows inside the window are removed
        "fade25_pct_per_100cyc": fade_per_100(g25),
        "cycles_to_soh80": below.min() if len(below) else np.nan,
        "soh_last": g.soh.iloc[-1],         # outliers removed here, unlike battery_summary.csv
        "T_max_med": g.T_max_C.median(),
    })


def build_fade(clean):
    rows = clean[clean.tier.isin(FADE_TIERS) & ~clean.cap_outlier & ~clean.break_in]
    rows = rows.sort_values(["battery_id", "cycle_valid"])
    fade = rows.groupby("battery_id").apply(fade_row, include_groups=False).reset_index()
    for c in ("n_used", "n_used25", "cycles_to_soh80"):
        fade[c] = fade[c].astype("Int64")
    return fade


def check(fade):
    assert len(fade) == 23
    f = fade.set_index("battery_id")
    # condition means of docs/Introduction_of_dataset.md section 8 (full series, first 25 cycles)
    expected = {
        ("B0005", "B0006", "B0007", "B0018", "B0036"): (17.9, 14.4),
        ("B0033", "B0034"): (8.3, 20.1),
        ("B0025", "B0026", "B0027"): (10.9, 9.7),
        ("B0028",): (14.6, 13.3),
        ("B0029", "B0030", "B0031", "B0032"): (28.4, 27.8),
        ("B0045", "B0046", "B0047", "B0048"): (29.9, 64.8),
        ("B0053", "B0055", "B0056"): (10.0, 27.9),
    }
    for ids, (full, w25) in expected.items():
        got = f.loc[list(ids), ["fade_pct_per_100cyc", "fade25_pct_per_100cyc"]].mean().round(1).tolist()
        assert got == [full, w25], (ids, got)
    never = set(f.index[f.cycles_to_soh80.isna()])
    assert never == set(f.index[f.tier == "B"]) | {"B0056"}, never


def main():
    clean = pd.read_csv(DATA / "cycles_clean.csv")
    fade = build_fade(clean)
    check(fade)
    fade.to_csv(DATA / "battery_fade.csv", index=False)
    print(f"battery_fade {len(fade)} rows -> {DATA.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
