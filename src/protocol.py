"""Shared modelling protocol for the four algorithms (docs/Rec.md section 6, docs/Go.md GĐ4).

load_split()    tier-A rows without cap_outlier / break_in, split by battery; test rows carry is_eval
naive(test)     the mandatory persistence baseline, in the shared prediction format
evaluate(pred)  MAE, RMSE, MAPE, R² per model, scenario and test battery, plus the RUL error of E2

Every model writes reports/predictions/<model>.csv with the columns PRED_COLS and predicts
exactly the is_eval rows (the last 40% of each test battery's cycles), in both scenarios:
    E1  one step ahead: true values up to cycle t-1 may be used
    E2  multi-step from the origin: only true values before the first is_eval row may be used
The target is soh; evaluate() rejects a file whose y_true is not the soh of those rows.

Run from the repository root to write the naive baseline and print its scores:
    python src/protocol.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PRED = ROOT / "reports" / "predictions"

TRAIN_IDS = ["B0005", "B0006", "B0018", "B0033", "B0045", "B0046", "B0048", "B0056"]
TEST_IDS = ["B0007", "B0034", "B0047", "B0055"]
ORIGIN_FRACTION = 0.6
EOL_SOH = 0.8
PRED_COLS = ["model", "battery_id", "cycle_valid", "scenario", "y_true", "y_pred", "y_lower", "y_upper"]
KEYS = ["battery_id", "cycle_valid"]


def load_split():
    """Return (train, test), sorted by battery and cycle_valid. Use train.battery_id as GroupKFold groups."""
    clean = pd.read_csv(DATA / "cycles_clean.csv", parse_dates=["start_time", "chg_end_time"])
    rows = clean[(clean.tier == "A") & ~clean.cap_outlier & ~clean.break_in]
    rows = rows.sort_values(KEYS).reset_index(drop=True)
    train = rows[rows.battery_id.isin(TRAIN_IDS)].reset_index(drop=True)
    test = rows[rows.battery_id.isin(TEST_IDS)].reset_index(drop=True)
    pos = test.groupby("battery_id").cumcount()
    n = test.groupby("battery_id").battery_id.transform("size")
    test["is_eval"] = pos >= (n * ORIGIN_FRACTION).astype(int)
    return train, test


def naive(test):
    """Persistence: E1 predicts the previous true soh; E2 holds the last soh known at the origin."""
    prev = test.groupby("battery_id").soh.shift()
    held = test.soh.where(~test.is_eval).groupby(test.battery_id).transform("last")
    ev = test[test.is_eval]
    parts = []
    for scenario, y_pred in (("E1", prev), ("E2", held)):
        parts.append(ev[KEYS].assign(model="naive", scenario=scenario, y_true=ev.soh, y_pred=y_pred[test.is_eval],
                                     y_lower=np.nan, y_upper=np.nan)[PRED_COLS])
    return pd.concat(parts, ignore_index=True)


def check_pred(pred, test):
    """Every (model, scenario) must predict soh on exactly the is_eval rows, so all models are scored alike."""
    missing = set(PRED_COLS) - set(pred.columns)
    assert not missing, f"missing columns {missing}"
    assert pred.scenario.isin(["E1", "E2"]).all(), "scenario must be E1 or E2"
    assert pred.y_pred.notna().all(), "y_pred has NaN"
    want = test.loc[test.is_eval, KEYS + ["soh"]].reset_index(drop=True)
    for (model, scenario), g in pred.groupby(["model", "scenario"]):
        g = g.sort_values(KEYS).reset_index(drop=True)
        assert g[KEYS].equals(want[KEYS]), f"{model} {scenario}: rows differ from the is_eval rows"
        assert np.allclose(g.y_true, want.soh), f"{model} {scenario}: y_true is not soh"


def scores(g):
    err = g.y_pred - g.y_true
    return pd.Series({
        "n": len(g),
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE_pct": (err.abs() / g.y_true).mean() * 100,
        "R2": 1 - (err ** 2).sum() / ((g.y_true - g.y_true.mean()) ** 2).sum(),
    })


def first_below(cycles, soh):
    """First cycle_valid with soh < EOL_SOH (same rule as cycles_to_soh80), NaN if never reached."""
    hit = cycles[soh < EOL_SOH]
    return hit.min() if len(hit) else np.nan


def evaluate(pred):
    """Return (metrics, rul). metrics has one row per model, scenario and battery_id ('all' = pooled)."""
    _, test = load_split()
    check_pred(pred, test)
    per = pred.groupby(["model", "scenario", "battery_id"]).apply(scores, include_groups=False).reset_index()
    pooled = pred.groupby(["model", "scenario"]).apply(scores, include_groups=False).reset_index()
    metrics = pd.concat([per, pooled.assign(battery_id="all")], ignore_index=True)
    metrics["n"] = metrics.n.astype(int)

    # RUL only for batteries still above EOL before the origin; B0047 crosses 0.8 before it
    known = test[~test.is_eval]
    before = known.groupby("battery_id").apply(lambda g: first_below(g.cycle_valid, g.soh), include_groups=False)
    e2 = pred[(pred.scenario == "E2") & pred.battery_id.isin(before.index[before.isna()])]
    rul = e2.groupby(["model", "battery_id"]).apply(lambda g: pd.Series({
        "eol_true": first_below(g.cycle_valid, g.y_true),
        "eol_pred": first_below(g.cycle_valid, g.y_pred),   # NaN: the forecast never reaches EOL
    }), include_groups=False).reset_index()
    rul["rul_error"] = rul.eol_pred - rul.eol_true          # > 0: the model warns too late
    return metrics, rul


def check(train, test, pred, rul):
    assert set(train.battery_id) == set(TRAIN_IDS) and set(test.battery_id) == set(TEST_IDS)
    assert (len(train), len(test), test.is_eval.sum()) == (947, 528, 213)
    assert set(rul.battery_id) == {"B0007", "B0034", "B0055"}
    for broken in (pred.iloc[1:], pred.assign(y_true=pred.y_true * 1.8)):   # a missing row; capacity in Ah
        try:
            evaluate(broken)
        except AssertionError:
            continue
        raise AssertionError("evaluate accepted an invalid prediction file")


def main():
    train, test = load_split()
    pred = naive(test)
    metrics, rul = evaluate(pred)
    check(train, test, pred, rul)
    pred.to_csv(PRED / "naive.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    print(rul.to_string(index=False))
    print(f"naive {len(pred)} rows -> {PRED.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
