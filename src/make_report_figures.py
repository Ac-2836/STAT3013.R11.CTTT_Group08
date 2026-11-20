"""Dataset and preprocessing figures for Report §3.1 (fig1-fig6), drawn from data/processed/.

Run: python src/make_report_figures.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# Ambient colours follow the validated categorical order: cold = blue, room = aqua, hot = orange.
AMBIENT = [("4°C", lambda t: t < 10, "#2a78d6"), ("22-24°C", lambda t: (t >= 20) & (t < 30), "#1baf7a"),
           ("43-44°C", lambda t: t >= 40, "#eb6834")]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10})


def ambient_color(t):
    return next(c for _, rule, c in AMBIENT if rule(t))


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_soh_curves(c):
    """Single-condition constant-current cells, SOH by valid cycle, one panel per discharge current."""
    cc = c[(c.profile == "constant_current") & ~c.cap_outlier & ~c.break_in & (c.tier != "N")]
    single = cc.groupby("battery_id").segment.nunique().loc[lambda s: s == 1].index
    cc = cc[cc.battery_id.isin(single)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, amp in zip(axes, [1.0, 2.0, 4.0]):
        for bid, g in cc[cc.I_level_A == amp].groupby("battery_id"):
            color = ambient_color(g.ambient_C.iloc[0])
            ax.plot(g.cycle_valid, g.soh, color=color, lw=1.4)
            if len(g) >= 60:   # short G3 runs overlap; their colour already identifies them
                ax.text(g.cycle_valid.iloc[-1] + 1, g.soh.iloc[-1], bid[1:], fontsize=7.5, color=MUTED, va="center")
        ax.axhline(0.8, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.set_title(f"Discharge {amp:g} A", loc="left", fontsize=10)
        ax.set_xlabel("Valid discharge cycle")
    axes[0].set_ylabel("SOH (C / C0)")
    handles = [plt.Line2D([], [], color=col, lw=2, label=lab) for lab, _, col in AMBIENT]
    fig.legend(handles=handles, title="Ambient", loc="upper right", ncol=3, frameon=False)
    fig.suptitle("SOH vs cycle, single-condition constant-current cells, outlier and break-in cycles removed "
                 "(dashed: SOH 0.8)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig1_soh_curves.png")


def fig2_capacity_definition(c):
    g = c[c.battery_id == "B0042"].sort_values("cycle")
    cold4 = g[(g.ambient_C == 4) & (g.I_level_A == 4)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(g.cycle, g.capacity_int_Ah, color="#2a78d6", lw=1.6,
            label=f"Integrated to own cutoff ({g.discharge_cutoff_V.iloc[0]:g} V)")
    ax.plot(g.cycle, g.capacity_Ah, color="#eb6834", lw=1.6, label="Raw 'Capacity' field (to 2.7 V)")
    for _, s in g.groupby("segment"):
        x = s.cycle.iloc[0]
        ax.axvline(x, color=MUTED, lw=0.8)
        ax.text(x + 0.8, 1.93, f"{s.ambient_C.iloc[0]:g}°C / {s.I_level_A.iloc[0]:g} A", fontsize=9, color=MUTED, va="top")
    ax.set(ylim=(0, 2.0), xlabel="Discharge cycle", ylabel="Capacity (Ah)")
    ax.set_title(f"B0042 at 4°C / 4 A: raw field reads ~{cold4.capacity_Ah.median():.2f} Ah, "
                 f"the cell still gives ~{cold4.capacity_int_Ah.median():.2f} Ah to 2.2 V", fontsize=11)
    ax.legend(loc="lower left", frameon=False)
    save(fig, "fig2_capacity_definition.png")
    return cold4.capacity_Ah.median(), cold4.capacity_int_Ah.median()


def fig3_rest_recovery(c):
    """Capacity change from the previous valid discharge of the same cell and condition, by rest length."""
    d = c[~c.cap_outlier].sort_values(["battery_id", "segment", "cycle_valid"]).copy()
    d["delta"] = d.groupby(["battery_id", "segment"]).capacity_int_Ah.diff()
    d = d.dropna(subset=["delta"])
    bins, labels = [0, 0.1, 1, 24, np.inf], ["< 6 min", "6 min - 1 h", "1 - 24 h", "> 1 day"]
    d["rest"] = pd.cut(d.rest_before_h, bins, labels=labels, right=False)
    groups = [d.loc[d.rest == lab, "delta"] for lab in labels]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.boxplot(groups, widths=0.5, showfliers=False, medianprops=dict(color="#2a78d6", lw=2),
               boxprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK))
    ax.set_xticks(range(1, 5), [f"{lab}\n(n = {len(x)})" for lab, x in zip(labels, groups)])
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set(xlabel="Rest before discharge", ylabel="Δ capacity vs previous cycle (Ah)")
    ax.set_title("Capacity temporarily recovers after long rests", loc="left", fontsize=11)
    save(fig, "fig3_rest_recovery.png")
    return {lab: x.median() for lab, x in zip(labels, groups)}


def fig4_condition_capacity(c):
    d = c[c.battery_id.isin(["B0038", "B0042"])]
    med = d.groupby(["battery_id", "segment"]).agg(cap=("capacity_int_Ah", "median"), amb=("ambient_C", "first"),
                                                  amp=("I_level_A", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(med))
    ax.bar(x, med.cap, width=0.6, color=[ambient_color(t) for t in med.amb])
    for xi, v in zip(x, med.cap):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{b}\n{a:g}°C {i:g} A" for b, a, i in zip(med.battery_id, med.amb, med.amp)])
    ax.set_ylabel("Median capacity (Ah)")
    ax.set_title("Same cell, different condition: capacity is partly reversible (cutoff 2.2 V)", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_condition_capacity.png")
    return med


def fig5_impedance_soh(c):
    g = c[(c.group == "G1") & ~c.cap_outlier].dropna(subset=["Re_ohm", "Rct_ohm"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, lab in zip(axes, ["Re_ohm", "Rct_ohm"], ["Re (Ω)", "Rct (Ω)"]):
        ax.scatter(g[col], g.soh, s=14, color="#2a78d6", alpha=0.6, edgecolors="white", linewidths=0.4)
        r = np.corrcoef(g[col], g.soh)[0, 1]
        ax.set(xlabel=lab)
        ax.text(0.03, 0.05, f"r = {r:.2f}, n = {len(g)}", transform=ax.transAxes, ha="left", fontsize=9, color=MUTED)
    axes[0].set_ylabel("SOH")
    fig.suptitle("G1 (24°C, 2 A): SOH falls as internal resistance rises", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_impedance_soh.png")


def fig6_distributions(c, fade):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.hist(fade.fade_pct_per_100cyc, bins=np.arange(4, 36, 2), color="#2a78d6", edgecolor="white")
    a1.set(xlabel="SOH loss per 100 cycles (%-points)", ylabel="Cells")
    a1.set_title(f"Fade rate across cells (n = {len(fade)})", loc="left", fontsize=11)
    a1.grid(axis="x", visible=False)
    groups = [(lab, c.loc[rule(c.ambient_C), "T_max_C"].dropna(), col) for lab, rule, col in AMBIENT]
    bp = a2.boxplot([g for _, g, _ in groups], widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK))
    for box, (_, _, col) in zip(bp["boxes"], groups):
        box.set(facecolor=col, alpha=0.5, edgecolor=INK)
    a2.set_xticks(range(1, 4), [lab for lab, _, _ in groups])
    a2.set(xlabel="Ambient", ylabel="Max cell temperature in discharge (°C)")
    a2.set_title("Self-heating during discharge", loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, "fig6_distributions.png")


def main():
    c = pd.read_csv(DATA / "cycles_clean.csv")
    fade = pd.read_csv(DATA / "battery_fade.csv")
    assert len(c) == 2658 and len(fade) == 23, "processed data differ from the locked pipeline"
    fig1_soh_curves(c)
    raw, integ = fig2_capacity_definition(c)
    assert raw < 0.2 and integ > 1.0, "B0042 4 A: raw field should be far below the integrated capacity"
    rest = fig3_rest_recovery(c)
    med = fig4_condition_capacity(c)
    fig5_impedance_soh(c)
    fig6_distributions(c, fade)
    print(f"B0042 4°C/4 A median: raw {raw:.3f} Ah, integrated {integ:.3f} Ah")
    print("median Δ capacity by rest:", {k: round(v, 4) for k, v in rest.items()})
    print(med.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
