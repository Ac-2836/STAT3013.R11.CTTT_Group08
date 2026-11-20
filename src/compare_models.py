"""Group model comparison (GĐ7) and EOL-alarm sensitivity (C4.1).

Reads every reports/predictions/*.csv, scores them with protocol.evaluate, and derives the
alarm "battery reaches SOH < 0.8 within the next H cycles" from each model's E2 trajectory.
Run: python src/compare_models.py
"""
import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import protocol as P

matplotlib.use("Agg")
TABLES = P.ROOT / "reports" / "tables"
FIGURES = P.ROOT / "reports" / "figures"
MODELS = ["naive", "regression", "arima", "arimax", "svr", "lstm"]
# Validated categorical order (dataviz reference palette); naive is the gray baseline.
COLORS = {"naive": "#8a8984", "regression": "#2a78d6", "arima": "#eb6834", "arimax": "#1baf7a",
          "svr": "#eda100", "lstm": "#e87ba4"}
STYLES = {"naive": (0, (4, 3)), "regression": "-", "arima": "-", "arimax": (0, (6, 2)),
          "svr": (0, (2, 1.5)), "lstm": "-"}
HORIZON = 20


def load_predictions():
    pred = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(P.PRED / "*.csv")))], ignore_index=True)
    assert set(pred.model) == set(MODELS), f"unexpected models: {sorted(set(pred.model))}"
    return pred


def coverage(pred):
    """Share of rows whose y_true lies inside [y_lower, y_upper], for model/scenario pairs that have intervals."""
    g = pred.dropna(subset=["y_lower", "y_upper"])
    inside = (g.y_true >= g.y_lower) & (g.y_true <= g.y_upper)
    return inside.groupby([g.model, g.scenario]).mean().rename("PI_coverage").reset_index()


def alarm_rows(pred, test, horizon=HORIZON):
    """One row per (model, decision cycle t) on the E2 trajectory of batteries still above EOL at the origin.

    truth: first observed cycle with SOH < 0.8 falls in (t, t + horizon].
    alarm: the first predicted crossing is at or before t + horizon (a crossing already predicted counts).
    Rows at or after the true crossing are dropped (event already happened), and so are rows whose window
    runs past the last observed cycle without an observed crossing (label unknown).
    """
    out = []
    for bid, g in test.groupby("battery_id"):
        known, ev = g[~g.is_eval], g[g.is_eval]
        if not np.isnan(P.first_below(known.cycle_valid, known.soh)):
            continue                                     # B0047: crossed before the origin (Rec.md 6.1 #3)
        eol_true = P.first_below(ev.cycle_valid, ev.soh)
        last = ev.cycle_valid.max()
        t = ev.cycle_valid.to_numpy()
        keep = t < eol_true if not np.isnan(eol_true) else t + horizon <= last
        truth = (eol_true > t) & (eol_true <= t + horizon)
        for model, p in pred[(pred.scenario == "E2") & (pred.battery_id == bid)].groupby("model"):
            p = p.sort_values("cycle_valid")
            assert np.array_equal(p.cycle_valid.to_numpy(), t)
            eol_pred = P.first_below(p.cycle_valid, p.y_pred)
            alarm = eol_pred <= t + horizon              # False when the forecast never crosses (NaN)
            out.append(pd.DataFrame({"model": model, "battery_id": bid, "cycle_valid": t, "truth": truth,
                                     "alarm": alarm})[keep])
    return pd.concat(out, ignore_index=True)


def confusion(rows):
    def one(g):
        tp = int((g.alarm & g.truth).sum()); fn = int((~g.alarm & g.truth).sum())
        fp = int((g.alarm & ~g.truth).sum()); tn = int((~g.alarm & ~g.truth).sum())
        return pd.Series({"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                          "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                          "specificity": tn / (tn + fp) if tn + fp else np.nan,
                          "precision": tp / (tp + fp) if tp + fp else np.nan,
                          "accuracy": (tp + tn) / len(g)})
    out = rows.groupby("model").apply(one, include_groups=False).reindex(MODELS)
    return out.astype({c: int for c in ["TP", "FN", "FP", "TN"]}).reset_index()


def plot_e2(pred, test, rul):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, (bid, g) in zip(axes.flat, test.groupby("battery_id")):
        ax.plot(g.cycle_valid, g.soh, color="#0b0b0b", lw=1.6, label="Observed SOH")
        for m in MODELS:
            p = pred[(pred.model == m) & (pred.scenario == "E2") & (pred.battery_id == bid)]
            ax.plot(p.cycle_valid, p.y_pred, color=COLORS[m], ls=STYLES[m], lw=1.6, label=m)
        ax.axhline(P.EOL_SOH, color="#52514e", lw=0.8)
        ax.axvline(g[g.is_eval].cycle_valid.min(), color="#52514e", lw=0.8, ls=":")
        ax.text(g.cycle_valid.min(), P.EOL_SOH, " EOL 0.8", va="bottom", fontsize=8, color="#52514e")
        ax.set_title(bid + (" (crossed 0.8 before origin)" if bid not in set(rul.battery_id) else ""), fontsize=10)
        ax.set_xlabel("Valid cycle"); ax.set_ylabel("SOH")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle("E2: multi-step forecasts from the 60% origin (dotted line) on the four test cells", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(FIGURES / "compare_e2_forecasts.png", dpi=200)
    plt.close(fig)


def plot_mae(pooled):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, scenario in zip(axes, ["E1", "E2"]):
        s = pooled[pooled.scenario == scenario].set_index("model").reindex(MODELS)
        y = np.arange(len(MODELS))
        ax.barh(y, s.MAE, height=0.6, color=[COLORS[m] for m in MODELS])
        ax.axvline(s.loc["naive", "MAE"], color="#52514e", lw=0.8, ls=(0, (4, 3)))
        for yi, v in zip(y, s.MAE):
            ax.text(v, yi, f" {v:.4f}", va="center", fontsize=8, color="#0b0b0b")
        ax.set_yticks(y, MODELS); ax.invert_yaxis()
        ax.set_title(f"{scenario} pooled MAE (213 rows; dashed = naive)", fontsize=10)
        ax.set_xlim(0, s.MAE.max() * 1.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "compare_mae.png", dpi=200)
    plt.close(fig)


def main():
    _, test = P.load_split()
    pred = load_predictions()
    metrics, rul = P.evaluate(pred)
    pooled = metrics[metrics.battery_id == "all"].drop(columns="battery_id")
    pooled = pooled.merge(coverage(pred), on=["model", "scenario"], how="left")
    pooled["model"] = pd.Categorical(pooled.model, MODELS)
    pooled = pooled.sort_values(["scenario", "model"])
    per_cell = metrics[metrics.battery_id != "all"].sort_values(["scenario", "battery_id", "model"])

    rows = alarm_rows(pred, test)
    alarm = confusion(rows)
    horizons = pd.concat([confusion(alarm_rows(pred, test, h)).assign(horizon=h) for h in (10, 20, 30)])

    # Self-checks: one label definition for all models; naive never crosses so it never alarms.
    assert rows.groupby("model").size().nunique() == 1
    assert rows.groupby(["battery_id", "cycle_valid"]).truth.nunique().max() == 1
    assert alarm.set_index("model").loc["naive", ["TP", "FP"]].sum() == 0
    assert len(pooled) == 2 * len(MODELS)

    TABLES.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(TABLES / "compare_metrics_pooled.csv", index=False)
    per_cell.to_csv(TABLES / "compare_metrics_by_cell.csv", index=False)
    rul.to_csv(TABLES / "compare_rul.csv", index=False)
    rows.to_csv(TABLES / "compare_alarm_rows.csv", index=False)
    alarm.to_csv(TABLES / "compare_alarm_h20.csv", index=False)
    horizons.to_csv(TABLES / "compare_alarm_horizons.csv", index=False)
    plot_e2(pred, test, rul)
    plot_mae(pooled)
    return dict(pooled=pooled, per_cell=per_cell, rul=rul, alarm=alarm, horizons=horizons, rows=rows)


if __name__ == "__main__":
    out = main()
    pd.set_option("display.width", 160)
    print(out["pooled"].round(4).to_string(index=False))
    print(out["rul"].to_string(index=False))
    print(out["alarm"].round(3).to_string(index=False))
