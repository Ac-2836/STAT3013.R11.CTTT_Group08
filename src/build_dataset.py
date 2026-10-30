"""Parse the NASA PCoE .mat files into tables and join them (preprocessing steps 1-5).

Reads   data/raw/B*.mat and data/raw/metadata.csv
Writes  data/processed/discharge_ops.csv   one row per discharge (2,794)
        data/processed/charge_ops.csv      one row per charge (2,815)
        data/processed/impedance_ops.csv   one row per EIS measurement (1,956)
        data/processed/cycles.csv          one row per discharge, joined with its
                                           preceding charge, EIS and metadata (not cleaned)

All rules follow docs/Rec.md section 2. Run from the repository root:
    python src/build_dataset.py
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

LOAD_A = 0.1      # |current| above this counts as a loaded sample
CC_END_V = 4.2    # end of the constant-current charge phase


def to_datetime(v):
    """MATLAB datevec [Y M D h m s] -> datetime, keeping fractional seconds."""
    y, mo, d, h, mi, s = v
    return datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=float(s))


def arr(x):
    return np.atleast_1d(x).astype(float)


def ah(y, t):
    """Trapezoid integral of current (A) over time (s), in Ah."""
    return float(np.trapezoid(y, t)) / 3600 if len(t) > 1 else 0.0


def discharge_row(d):
    t, i, v, temp = arr(d["Time"]), arr(d["Current_measured"]), arr(d["Voltage_measured"]), arr(d["Temperature_measured"])
    load = i < -LOAD_A
    y = np.where(load, -i, 0.0)                 # discharge current made positive, unloaded samples = 0
    below = np.flatnonzero(v < 2.7)
    k = below[0] + 1 if below.size else len(v)  # up to and including the first sample below 2.7 V
    cap = np.atleast_1d(d["Capacity"])          # empty for some G8 discharges
    i_mean = float(-i[load].mean()) if load.any() else np.nan
    return {
        "n_samples": len(t),
        "capacity_Ah": float(cap[0]) if cap.size else np.nan,
        "capacity_to_2p7V_Ah": ah(y[:k], t[:k]),
        "capacity_int_Ah": ah(y, t),
        "I_load_mean_A": i_mean,
        "I_level_A": float(np.round(i_mean)),
        "V_min": float(v.min()),
        "duration_s": float(t[-1]),
        "t_to_cutoff_s": float(t[np.flatnonzero(load)[-1]]) if load.any() else np.nan,
        "energy_Wh": ah(v * y, t),
        "T_start_C": float(temp[0]),
        "T_max_C": float(temp.max()),
        "T_mean_C": float(temp.mean()),
    }


def charge_row(d, start):
    t, i, v, temp = arr(d["Time"]), arr(d["Current_measured"]), arr(d["Voltage_measured"]), arr(d["Temperature_measured"])
    hit = np.flatnonzero(v >= CC_END_V)
    return {
        "chg_duration_s": float(t[-1]),
        "chg_end_time": start + timedelta(seconds=float(t[-1])),
        "chg_charge_Ah": ah(np.where(i > LOAD_A, i, 0.0), t),  # skips the start-up noise (e.g. -4 A)
        "chg_cc_time_s": float(t[hit[0]]) if hit.size else np.nan,
        "chg_T_max_C": float(temp.max()),
    }


def resistance(x):
    """Re/Rct as a float; a complex value with a nonzero imaginary part is a failed fit -> NaN."""
    z = np.atleast_1d(x)
    if not z.size or np.imag(z[0]) != 0:
        return np.nan
    return float(np.real(z[0]))


def parse_all():
    dis, chg, eis = [], [], []
    for f in sorted(RAW.glob("B*.mat")):
        bid = f.stem
        n_dis = 0
        for op_index, op in enumerate(loadmat(f, simplify_cells=True)[bid]["cycle"]):
            base = {
                "battery_id": bid,
                "op_index": op_index,
                "ambient_C": float(op["ambient_temperature"]),
                "start_time": to_datetime(op["time"]),
            }
            d = op["data"]
            if op["type"] == "discharge":
                n_dis += 1
                dis.append({**base, "cycle": n_dis, **discharge_row(d)})
            elif op["type"] == "charge":
                chg.append({**base, **charge_row(d, base["start_time"])})
            elif op["type"] == "impedance":
                eis.append({**base, "Re_ohm": resistance(d["Re"]), "Rct_ohm": resistance(d["Rct"])})
            else:
                raise ValueError(f"{bid} op {op_index}: unknown type {op['type']!r}")
    return pd.DataFrame(dis), pd.DataFrame(chg), pd.DataFrame(eis)


def charges_since_previous_discharge(cycles, chg):
    """Number of charges between the previous discharge and this one, per battery."""
    out = pd.Series(0, index=cycles.index)
    for bid, g in cycles.groupby("battery_id"):
        c = np.sort(chg.loc[chg.battery_id == bid, "op_index"].to_numpy())
        ops = g.sort_values("op_index").op_index.to_numpy()
        prev = np.concatenate([[-1], ops[:-1]])
        out[g.sort_values("op_index").index] = np.searchsorted(c, ops) - np.searchsorted(c, prev, side="right")
    return out


def join(dis, chg, eis):
    meta = pd.read_csv(RAW / "metadata.csv")
    meta["profile"] = np.where(meta.discharge_protocol.str.contains("Square Wave"), "square_wave_0.05Hz", "constant_current")
    meta = meta[["battery_id", "group", "discharge_cutoff_V", "profile"]]

    # merge_asof needs the key sorted over the whole table, not within each battery
    left = dis.sort_values("op_index")
    right_chg = (chg.drop(columns=["ambient_C", "start_time"])
                 .rename(columns={"op_index": "chg_op_index"}).sort_values("chg_op_index"))
    right_eis = (eis[["battery_id", "op_index", "Re_ohm", "Rct_ohm"]]
                 .rename(columns={"op_index": "eis_op_index"}).sort_values("eis_op_index"))
    asof = dict(by="battery_id", direction="backward", allow_exact_matches=False)
    cycles = pd.merge_asof(left, right_chg, left_on="op_index", right_on="chg_op_index", **asof)
    cycles = pd.merge_asof(cycles, right_eis, left_on="op_index", right_on="eis_op_index", **asof)

    cycles["rest_before_h"] = (cycles.start_time - cycles.chg_end_time).dt.total_seconds() / 3600
    cycles["n_chg_since_prev_dis"] = charges_since_previous_discharge(cycles, chg)
    cycles = cycles.merge(meta, on="battery_id", how="left", validate="many_to_one")
    cycles = cycles.sort_values(["battery_id", "op_index"]).reset_index(drop=True)

    cols = ["battery_id", "op_index", "cycle", "ambient_C", "start_time", "n_samples",
            "capacity_Ah", "capacity_to_2p7V_Ah", "capacity_int_Ah", "I_load_mean_A", "I_level_A", "V_min",
            "duration_s", "t_to_cutoff_s", "energy_Wh", "T_start_C", "T_max_C", "T_mean_C",
            "group", "discharge_cutoff_V", "profile",
            "chg_op_index", "chg_duration_s", "chg_end_time", "chg_charge_Ah", "chg_cc_time_s", "chg_T_max_C",
            "rest_before_h", "n_chg_since_prev_dis",
            "eis_op_index", "Re_ohm", "Rct_ohm"]
    return cycles[cols]


def check(dis, chg, eis, cycles):
    assert (len(dis), len(chg), len(eis)) == (2794, 2815, 1956)
    assert dis.battery_id.nunique() == 34
    assert len(cycles) == 2794
    assert cycles.group.notna().all()
    has_chg, has_eis = cycles.chg_op_index.notna(), cycles.eis_op_index.notna()
    assert (cycles.chg_op_index[has_chg] < cycles.op_index[has_chg]).all()   # charge always before the discharge
    assert (cycles.eis_op_index[has_eis] < cycles.op_index[has_eis]).all()   # EIS never taken from the future
    assert (cycles.rest_before_h.dropna() >= 0).all()
    assert (~has_chg).sum() == 23 and (~has_eis).sum() == 76
    for name, df in (("discharge", dis), ("charge", chg), ("impedance", eis)):
        for _, g in df.groupby("battery_id"):
            assert g.op_index.is_monotonic_increasing, name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dis, chg, eis = parse_all()
    cycles = join(dis, chg, eis)
    check(dis, chg, eis, cycles)
    dis.to_csv(OUT / "discharge_ops.csv", index=False)
    chg.to_csv(OUT / "charge_ops.csv", index=False)
    eis.to_csv(OUT / "impedance_ops.csv", index=False)
    cycles.to_csv(OUT / "cycles.csv", index=False)
    print(f"discharge_ops {len(dis)}, charge_ops {len(chg)}, impedance_ops {len(eis)}, cycles {len(cycles)} rows -> {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
