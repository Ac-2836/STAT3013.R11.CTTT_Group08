"""Descriptive statistics (Minh Vu, branch minh-vu). Go.md stage 2, WBS C1.5–C1.7, slides Chap 1 §3–4.

Classes, each with one job:

    DescConfig          Shared settings: tiers used, variable lists, colours, figure folder.
    DescriptiveData     Reads 3 processed tables, filters rows, adds temperature and condition labels.
    MeasurementScales   Scale of measurement (nominal/ordinal/interval/ratio) of every column, with the reason.
    SummaryStats        Table of descriptive measures (n, mean, median, Q1, Q3, SD, SE, min, max, range,
                        skewness, kurtosis) for a whole table or split by group.
    DescriptivePlots    6 figures following Go.md 2.2.
    InsightWriter       Writes the insight sentences; every number in them is computed from the data.
    DescriptiveAnalysis Calls the classes above in order.

Two levels of analysis:
    - CYCLE level:   one row per discharge (cycles_clean.csv). Used to describe distributions.
    - BATTERY level: one row per battery (battery_fade.csv, battery_summary.csv). Cycles of one battery
                     are strongly dependent (F9), so comparisons between conditions use the battery level.

Run from the repository root:
    python src/descriptive_stats.py
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


# ----------------------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------------------
@dataclass
class DescConfig:
    # Tiers A, B, N: batteries whose conditions do NOT change (tier C changes mid-way, X is broken)
    tiers: tuple = ("A", "B", "N")

    # Variables described at the cycle level and at the battery level (Go.md 2.1)
    cycle_vars: tuple = ("capacity_int_Ah", "soh", "T_max_C", "Re_ohm", "Rct_ohm", "rest_before_h")
    battery_vars: tuple = ("C0_Ah", "soh_last", "fade_pct_per_100cyc", "fade25_pct_per_100cyc", "cycles_to_soh80")
    # Cycle-level variables also described per operating condition (C1.5)
    condition_vars: tuple = ("soh", "T_max_C", "rest_before_h")

    # Colours by ambient temperature: first 3 colours of the standard palette (colour-blind safe)
    temp_colors: dict = field(default_factory=lambda: {"4°C": "#2a78d6", "24°C": "#1baf7a", "43°C": "#eb6834"})
    temp_order: tuple = ("4°C", "24°C", "43°C")
    ink: str = "#52514e"                    # secondary text colour, reference lines

    eol_soh: float = 0.8
    long_rest_h: float = 1.0                # threshold of the long_rest flag (Rec.md section 2, row 9)
    fig_dir: Path = ROOT / "reports" / "figures"


# ----------------------------------------------------------------------------------------
# 2. Data
# ----------------------------------------------------------------------------------------
class DescriptiveData:
    """Reads the data and prepares labels.

    cycles  : cycles_clean.csv, tiers A/B/N only, WITHOUT the cap_outlier and break_in rows
              (same as Introduction_of_dataset section 8, n = 2,065).
    fade    : battery_fade.csv, 23 batteries (tiers A, B, N).
    summary : battery_summary.csv, tier A/B/N batteries only (for C0_Ah).
    Two added columns:
      temp      : "4°C", "24°C" or "43°C" (grouped from ambient_C)
      condition : operating condition = temperature × discharge current × profile, e.g. "24°C, 2 A, CC"
    """

    def __init__(self, config: DescConfig):
        self.cfg = config

    @staticmethod
    def temp_label(ambient_c: pd.Series) -> pd.Series:
        return pd.Series(np.select([ambient_c < 10, ambient_c < 30], ["4°C", "24°C"], "43°C"),
                         index=ambient_c.index)

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["temp"] = self.temp_label(df.ambient_C)
        kind = np.where(df.profile == "constant_current", "CC", "square wave")
        df["condition"] = df.temp + ", " + df.I_level_A.astype(int).astype(str) + " A, " + kind
        return df

    def load(self):
        clean = pd.read_csv(DATA / "cycles_clean.csv", parse_dates=["start_time", "chg_end_time"])
        keep = clean.tier.isin(self.cfg.tiers) & ~clean.cap_outlier & ~clean.break_in
        self.cycles = self.add_labels(clean[keep].reset_index(drop=True))
        self.all_clean = clean                                   # full table, to count the filtered rows

        self.fade = self.add_labels(pd.read_csv(DATA / "battery_fade.csv"))
        summary = pd.read_csv(DATA / "battery_summary.csv")
        self.summary = summary[summary.tier.isin(self.cfg.tiers)].reset_index(drop=True)

        # Battery-level table: add C0_Ah (from summary) to the fade table
        self.batteries = self.fade.merge(self.summary[["battery_id", "C0_Ah"]], on="battery_id", how="left")
        return self

    def condition_order(self) -> list:
        """Order of the condition groups in tables and figures: by temperature, then current."""
        c = self.batteries[["condition", "ambient_C", "I_level_A", "profile"]].drop_duplicates("condition")
        return c.sort_values(["ambient_C", "I_level_A", "profile"]).condition.tolist()


# ----------------------------------------------------------------------------------------
# 3. Descriptive measures
# ----------------------------------------------------------------------------------------
class SummaryStats:
    """Table of measures following the list on slide Chap 1, p. 57.

    Notes on skewness and kurtosis:
    - skewness: pandas .skew() (sample-adjusted). Negative = left-skewed (long tail towards small values).
    - kurtosis: pandas .kurt() returns EXCESS kurtosis (normal = 0). The slides use Pearson kurtosis
      (normal = 3), so 3 is ADDED (Go.md 2.1).
    - SE = SD / sqrt(n): standard error of the mean.
    """

    COLS = ["n", "n_missing", "mean", "median", "Q1", "Q3", "SD", "SE", "min", "max", "range",
            "skewness", "kurtosis"]

    @staticmethod
    def describe(x: pd.Series) -> pd.Series:
        n_missing = int(x.isna().sum())
        x = x.dropna()
        n = len(x)
        return pd.Series({
            "n": n, "n_missing": n_missing,
            "mean": x.mean(), "median": x.median(), "Q1": x.quantile(0.25), "Q3": x.quantile(0.75),
            "SD": x.std(), "SE": x.std() / np.sqrt(n) if n else np.nan,
            "min": x.min(), "max": x.max(), "range": x.max() - x.min(),
            "skewness": x.skew() if n > 2 else np.nan,
            "kurtosis": x.kurt() + 3 if n > 3 else np.nan,
        })[SummaryStats.COLS]

    def table(self, df: pd.DataFrame, variables) -> pd.DataFrame:
        """One row per variable."""
        return pd.DataFrame({v: self.describe(df[v]) for v in variables}).T

    def by_group(self, df: pd.DataFrame, variable: str, group: str, order=None) -> pd.DataFrame:
        """One variable split by group (e.g. T_max_C by temp). One row per group."""
        out = df.groupby(group)[variable].apply(self.describe).unstack()
        if order is not None:
            out = out.reindex([g for g in order if g in out.index])
        return out


# ----------------------------------------------------------------------------------------
# 3b. Measurement scales (WBS C1.1, Chap 1, pp. 22–32)
# ----------------------------------------------------------------------------------------
class MeasurementScales:
    """Scale of measurement of EVERY column in the 3 tables used here, with the reason.

    Follows the data dictionary (Introduction_of_dataset.md section 6). One deliberate difference:
    chg_end_time is a timestamp, so it is Interval (like start_time), not Ratio.
    check() fails if a column of the tables has no scale, so the table stays complete when columns are added.
    """

    NOM_ID = "label that identifies or groups; only counts, mode and frequency tables make sense"
    NOM_FLAG = "yes/no flag (bool); only counts and shares make sense"
    INTERVAL_T = "temperature in °C: differences are meaningful but 0°C is not 'no heat', so ratios are not"
    INTERVAL_TIME = "calendar timestamp: differences (durations) are meaningful, there is no true zero"
    RATIO_COUNT = "count or position: true zero, ratios meaningful (discrete)"
    RATIO_PHYS = "physical quantity with a true zero: all operations, including ratios, are valid"

    SCALES = {
        # identifiers and groups
        "battery_id": ("Nominal", NOM_ID), "group": ("Nominal", NOM_ID), "profile": ("Nominal", NOM_ID),
        "tier": ("Nominal", "quality class A/B/C/N/X; the letters are not ordered by a measured quantity"),
        "condition": ("Nominal", NOM_ID), "temp": ("Nominal", "label of the ambient group (4°C / 24°C / 43°C)"),
        # flags
        "cap_outlier": ("Nominal", NOM_FLAG), "break_in": ("Nominal", NOM_FLAG), "long_rest": ("Nominal", NOM_FLAG),
        # order without equal steps
        "segment": ("Ordinal", "1st, 2nd, ... block of constant conditions: ordered, but the steps are not equal amounts"),
        # time stamps and temperatures
        "start_time": ("Interval", INTERVAL_TIME), "chg_end_time": ("Interval", INTERVAL_TIME),
        "ambient_C": ("Interval", INTERVAL_T), "T_start_C": ("Interval", INTERVAL_T), "T_max_C": ("Interval", INTERVAL_T),
        "T_mean_C": ("Interval", INTERVAL_T), "chg_T_max_C": ("Interval", INTERVAL_T), "T_max_med": ("Interval", INTERVAL_T),
        # counts and positions
        "op_index": ("Ratio (discrete)", RATIO_COUNT), "chg_op_index": ("Ratio (discrete)", RATIO_COUNT),
        "eis_op_index": ("Ratio (discrete)", RATIO_COUNT), "cycle": ("Ratio (discrete)", RATIO_COUNT),
        "cycle_valid": ("Ratio (discrete)", RATIO_COUNT), "n_chg_since_prev_dis": ("Ratio (discrete)", RATIO_COUNT),
        "n_discharges_raw": ("Ratio (discrete)", RATIO_COUNT), "n_dropped": ("Ratio (discrete)", RATIO_COUNT),
        "n_segments": ("Ratio (discrete)", RATIO_COUNT), "n_valid": ("Ratio (discrete)", RATIO_COUNT),
        "n_cap_outlier": ("Ratio (discrete)", RATIO_COUNT), "n_break_in": ("Ratio (discrete)", RATIO_COUNT),
        "n_used": ("Ratio (discrete)", RATIO_COUNT), "n_used25": ("Ratio (discrete)", RATIO_COUNT),
        "cycles_to_soh80": ("Ratio (discrete)", "number of cycles until SOH < 0.8; missing (censored) if never reached"),
        # physical quantities
        "I_level_A": ("Ratio", RATIO_PHYS), "discharge_cutoff_V": ("Ratio", RATIO_PHYS),
        "capacity_Ah": ("Ratio", RATIO_PHYS), "capacity_to_2p7V_Ah": ("Ratio", RATIO_PHYS),
        "capacity_int_Ah": ("Ratio", RATIO_PHYS), "C0_Ah": ("Ratio", RATIO_PHYS),
        "soh": ("Ratio", "capacity / C0: 0 means no capacity left, so ratios are meaningful"),
        "soh_segment": ("Ratio", "capacity / C0 of the segment"), "soh_min": ("Ratio", "capacity / C0"),
        "soh_last": ("Ratio", "capacity / C0 (note: defined differently in summary and fade, see the dictionary)"),
        "I_load_mean_A": ("Ratio", RATIO_PHYS), "V_min": ("Ratio", RATIO_PHYS), "duration_s": ("Ratio", RATIO_PHYS),
        "t_to_cutoff_s": ("Ratio", RATIO_PHYS), "energy_Wh": ("Ratio", RATIO_PHYS), "chg_duration_s": ("Ratio", RATIO_PHYS),
        "chg_charge_Ah": ("Ratio", RATIO_PHYS), "chg_cc_time_s": ("Ratio", RATIO_PHYS),
        "rest_before_h": ("Ratio", RATIO_PHYS), "Re_ohm": ("Ratio", RATIO_PHYS), "Rct_ohm": ("Ratio", RATIO_PHYS),
        "outlier_share": ("Ratio", "share n_cap_outlier / n_valid, from 0 to 1"),
        "fade_pct_per_100cyc": ("Ratio", "rate (SOH points lost per 100 cycles); 0 = no degradation"),
        "fade25_pct_per_100cyc": ("Ratio", "rate over the first 25 cycles; 0 = no degradation"),
    }
    VALID_OPS = {"Nominal": "count, mode, frequency table", "Ordinal": "+ order, median, percentiles",
                 "Interval": "+ differences, mean, SD", "Ratio": "+ ratios (\"twice as large\")",
                 "Ratio (discrete)": "+ ratios (\"twice as large\")"}

    def table(self, data: "DescriptiveData") -> pd.DataFrame:
        """One row per column of cycles_clean.csv, battery_fade.csv and battery_summary.csv."""
        sources = {"cycles_clean": data.cycles, "battery_fade": data.fade, "battery_summary": data.summary}
        rows = {}
        for name, df in sources.items():
            for col in df.columns:
                rows.setdefault(col, []).append(name)
        missing = sorted(set(rows) - set(self.SCALES))
        assert not missing, f"columns without a scale: {missing}"
        out = pd.DataFrame([{"column": c, "tables": ", ".join(t), "scale": self.SCALES[c][0],
                             "valid operations": self.VALID_OPS[self.SCALES[c][0]], "reason": self.SCALES[c][1]}
                            for c, t in rows.items()])
        order = ["Nominal", "Ordinal", "Interval", "Ratio (discrete)", "Ratio"]
        return out.assign(_o=out.scale.map(order.index)).sort_values(["_o", "column"]).drop(columns="_o").reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------------------
def fmt_int(x) -> str:
    """Integer with a thousands separator: 2065 -> 2,065."""
    return f"{int(x):,}"


class DescriptivePlots:
    """6 figures (Go.md 2.2), saved in reports/figures/ with the prefix desc_."""

    def __init__(self, config: DescConfig, data: DescriptiveData):
        self.cfg = config
        self.data = data

    def _save(self, fig, name):
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / name, dpi=150)
        return fig

    @staticmethod
    def _clean_axes(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.8)
        ax.set_axisbelow(True)

    def soh_histogram(self):
        """Figure 1: cycle-level SOH distribution, with mean and median lines."""
        x = self.data.cycles.soh
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(x, bins=40, color=self.cfg.temp_colors["4°C"], edgecolor="white", linewidth=0.6)
        for val, name, ls in ((x.mean(), "mean", "-"), (x.median(), "median", "--")):
            ax.axvline(val, color=self.cfg.ink, ls=ls, lw=1.4)
            ax.text(val, ax.get_ylim()[1] * 0.97, f" {name} = {val:.3f}", color=self.cfg.ink, fontsize=9,
                    ha="right" if name == "mean" else "left", va="top")
        ax.axvline(self.cfg.eol_soh, color="0.6", ls=":", lw=1)
        ax.set_xlabel("SOH")
        ax.set_ylabel("number of cycles")
        ax.set_title(f"Cycle-level SOH distribution (tiers A/B/N, n = {fmt_int(len(x))}); dotted line = EOL 0.8")
        self._clean_axes(ax)
        return self._save(fig, "desc_soh_hist.png")

    def rest_histogram(self):
        """Figure 2: rest time before each discharge. Log x axis (extremely right-skewed), tick labels in
        minutes / hours / days for readability, with median, mean, the long_rest threshold and a stats box."""
        x = self.data.cycles.rest_before_h
        s = SummaryStats.describe(x)
        thr = self.cfg.long_rest_h
        n_long = int((x > thr).sum())
        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 45)
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        # Bars of rests <= 1 hour in a light colour, bars of rests > 1 hour (long_rest) in a strong colour
        _, edges, patches = ax.hist(x, bins=bins, edgecolor="white", linewidth=0.6)
        for p, left in zip(patches, edges[:-1]):
            p.set_facecolor("#d1495b" if left >= thr else "#8fb4e3")
        ax.set_xscale("log")
        # Everyday units on the x axis instead of 10^-1, 10^0 ...
        ticks = [1 / 60, 5 / 60, 0.5, 1, 6, 24, 24 * 7, 24 * 14]
        labels = ["1 min", "5 min", "30 min", "1 h", "6 h", "1 day", "1 week", "2 weeks"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
        ax.set_xlim(x.min() * 0.9, x.max() * 1.15)
        # Median, mean and long_rest threshold lines
        ax.axvline(thr, color=self.cfg.ink, ls="--", lw=1.2)
        ax.axvline(s["median"], color="#1f4e8c", lw=1.6)
        ax.axvline(s["mean"], color="#a02c3b", lw=1.6, ls=":")
        top = ax.get_ylim()[1]
        ax.text(s["median"] * 1.08, top * 0.97, f"median = {s['median'] * 60:.1f} min", color="#1f4e8c", fontsize=9, va="top")
        ax.text(s["mean"] * 0.93, top * 0.80, f"mean = {s['mean']:.2f} h", color="#a02c3b", fontsize=9, va="top", ha="right")
        ax.text(thr * 1.1, top * 0.62, f"long_rest threshold = 1 h\n{n_long} cycles rested longer (red)",
                color=self.cfg.ink, fontsize=9, va="top")
        # Stats box inside the figure
        box = (f"n = {fmt_int(s['n'])}\nmin = {s['min'] * 60:.1f} min\nmedian = {s['median'] * 60:.1f} min\n"
               f"mean = {s['mean']:.2f} h\nmax = {s['max']:.1f} h (≈ {s['max'] / 24:.0f} days)\n"
               f"skewness = {s['skewness']:.1f}\nkurtosis = {s['kurtosis']:.0f}")
        ax.text(0.985, 0.97, box, transform=ax.transAxes, ha="right", va="top", fontsize=9, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.8"))
        ax.set_xlabel("rest time before the discharge (log scale: ticks are not evenly spaced in time)")
        ax.set_ylabel("number of cycles per bar")
        ax.set_title("Rest before discharge: most rests are very short, a few are very long")
        self._clean_axes(ax)
        return self._save(fig, "desc_rest_hist.png")

    def tmax_boxplot(self):
        """Figure 3: maximum cell temperature during discharge, split by ambient temperature.
        Written on the figure: median, Q1–Q3, ambient level (dashed line) and how to read a box."""
        c = self.data.cycles
        order = [t for t in self.cfg.temp_order if t in set(c.temp)]
        amb = {"4°C": 4, "24°C": 24, "43°C": 43}
        fig, ax = plt.subplots(figsize=(10, 5.6))
        bp = ax.boxplot([c.T_max_C[c.temp == t] for t in order], patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "lw": 2}, flierprops={"markersize": 3, "alpha": 0.5})
        for patch, t in zip(bp["boxes"], order):
            patch.set_facecolor(self.cfg.temp_colors[t])
            patch.set_alpha(0.85)
        for i, t in enumerate(order, start=1):
            x = c.T_max_C[c.temp == t]
            q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
            # dashed line = ambient temperature (if the cell did not heat up, the box would sit here)
            ax.hlines(amb[t], i - 0.33, i + 0.33, colors=self.cfg.ink, linestyles="--", lw=1.4)
            ax.text(i + 0.36, amb[t], f"ambient {amb[t]}°C", va="center", fontsize=8.5, color=self.cfg.ink)
            ax.text(i - 0.30, med, f"median\n{med:.1f}°C", ha="right", va="center", fontsize=9, fontweight="bold")
            ax.text(i, x.max() + 1.2, f"box: {q1:.1f} to {q3:.1f}°C", ha="center", fontsize=8.5, color=self.cfg.ink)
        ax.set_xticks(range(1, len(order) + 1),
                      [f"{t}\n(n = {fmt_int((c.temp == t).sum())} cycles)" for t in order])
        ax.set_xlim(0.4, len(order) + 0.95)
        ax.set_ylim(0, c.T_max_C.max() + 6)
        # how to read a box
        guide = ("How to read a box:\n• black line = median (half of the cycles are lower)\n• box = middle 50% of cycles (Q1 to Q3)\n"
                 "• whiskers = the rest, except unusual points\n• single dots (if any) = unusually far points\n• dashed line = ambient temperature")
        ax.text(0.01, 0.99, guide, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.8"))
        ax.set_xlabel("ambient temperature of the test chamber")
        ax.set_ylabel("maximum cell temperature in the discharge, T_max_C (°C)")
        ax.set_title("Cells heat up during discharge: how much hotter than the ambient?")
        self._clean_axes(ax)
        return self._save(fig, "desc_tmax_box.png")

    def fade_by_condition(self):
        """Figure 4: degradation rate by operating condition, BATTERY level.

        Each group has only 1–5 batteries, so every battery is drawn as a dot and a horizontal line marks
        the mean (value written next to it), instead of a boxplot (misleading with 2–3 points).
        Two panels: whole series and first 25 cycles. Includes a reading guide and a temperature legend.
        """
        from matplotlib.lines import Line2D
        b = self.data.batteries
        order = self.data.condition_order()
        fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), sharey=True)
        for ax, var, title in zip(axes, ["fade_pct_per_100cyc", "fade25_pct_per_100cyc"],
                                  ["Left: over the whole life of each battery",
                                   "Right: first 25 cycles only (same window for every battery)"]):
            for i, cond in enumerate(order):
                g = b[b.condition == cond]
                color = self.cfg.temp_colors[g.temp.iloc[0]]
                jitter = np.linspace(-0.12, 0.12, len(g)) if len(g) > 1 else [0]
                normal = g[g.tier != "N"]
                ax.scatter(i + np.asarray(jitter)[: len(g)][(g.tier != "N").values], normal[var], s=55,
                           color=color, edgecolor="white", linewidth=0.8, zorder=3)
                noisy = g[g.tier == "N"]
                if len(noisy):                                   # B0054 (tier N, noisy): hollow dot
                    ax.scatter(i + np.asarray(jitter)[: len(g)][(g.tier == "N").values], noisy[var], s=55,
                               facecolor="white", edgecolor=color, linewidth=1.5, zorder=3)
                mean = normal[var].mean()
                ax.plot([i - 0.25, i + 0.25], [mean, mean], color="black", lw=1.8, zorder=4)
                ax.text(i + 0.27, mean, f"{mean:.1f}", va="center", fontsize=9.5, fontweight="bold")
            def n_label(o):
                g = b[b.condition == o]
                k = int((g.tier == "N").sum())
                return f"({len(g) - k}+{k} noisy)" if k else f"({len(g)} batt.)"
            def life_label(o):
                # Left panel: add the real number of cycles and the final SOH, to separate "rate" from "what is left"
                g = b[(b.condition == o) & (b.tier != "N")]
                return f"\n{g.n_used.mean():.0f} cycles\nfinal SOH {100 * g.soh_last.mean():.0f}%"
            extra = life_label if var == "fade_pct_per_100cyc" else (lambda o: "")
            ax.set_xticks(range(len(order)), [o.replace(", ", "\n", 1) + "\n" + n_label(o) + extra(o) for o in order], fontsize=8)
            ax.set_xlim(-0.6, len(order) - 0.2)
            ax.set_ylim(0, 90)
            ax.set_title(title, fontsize=11)
            self._clean_axes(ax)
        axes[0].set_ylabel("degradation rate: SOH points (%) lost per 100 cycles")
        guide = ("How to read:\n• each dot = 1 battery; black line + bold number = group mean\n"
                 "• HIGHER dot = battery ages faster\n"
                 "• e.g. 30 = at this pace, 100 cycles cost 30 points of SOH\n"
                 "• hollow dot = battery B0054 (noisy data), not in the mean\n"
                 "• last 2 lines under each group (left) = mean cycles run and SOH left at the end")
        axes[0].text(0.015, 0.985, guide, transform=axes[0].transAxes, va="top", ha="left", fontsize=8.8, linespacing=1.5,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.8"))
        handles = [Line2D([], [], marker="o", ls="", color=self.cfg.temp_colors[t], label=f"ambient {t}") for t in self.cfg.temp_order]
        axes[1].legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.suptitle("Which conditions make batteries age faster? Degradation rate by operating condition", fontsize=13)
        return self._save(fig, "desc_fade_by_condition.png")

    def soh_curves(self):
        """Figure 5: SOH curves over cycles. Panel = discharge current, colour = temperature, dashed = square wave.
        Black ✕ = first time the battery reaches EOL (SOH 0.8). Annotates the spikes (capacity recovery) and the noisy battery B0054."""
        c = self.data.cycles
        bat = self.data.batteries.set_index("battery_id")
        levels = sorted(c.I_level_A.unique())
        fig, axes = plt.subplots(1, len(levels), figsize=(5.6 * len(levels), 5.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, lvl in zip(axes, levels):
            sub = c[c.I_level_A == lvl]
            n_eol = 0
            for bid, g in sub.groupby("battery_id"):
                ls = "-" if g.profile.iloc[0] == "constant_current" else "--"
                ax.plot(g.cycle_valid, g.soh, color=self.cfg.temp_colors[g.temp.iloc[0]], lw=1.1, ls=ls, alpha=0.9)
                k = bat.loc[bid, "cycles_to_soh80"]
                if pd.notna(k):                                  # mark the EOL point
                    n_eol += 1
                    y = g.loc[g.cycle_valid == k, "soh"]
                    ax.scatter([k], [y.iloc[0] if len(y) else self.cfg.eol_soh], marker="x", s=36, color="black", zorder=5, linewidths=1.6)
            ax.axhline(self.cfg.eol_soh, color="0.6", ls=":", lw=1)
            ax.text(0.02, self.cfg.eol_soh - 0.008, "EOL = 0.8", transform=ax.get_yaxis_transform(), ha="left", va="top", fontsize=8.5, color="0.4")
            ax.set_title(f"Discharge current {lvl:.0f} A ({sub.battery_id.nunique()} batteries, {n_eol} reached EOL)", fontsize=11)
            ax.set_xlabel("cycle number (cycle_valid)")
            self._clean_axes(ax)
        axes[0].set_ylabel("SOH (1.0 = as new)")
        # Recovery spikes in the 1 A panel: use the actual cycles with a long rest in the data
        sub1 = c[c.I_level_A == levels[0]]
        lr = sub1[sub1.rest_before_h > 24]
        n_pin = sub1.battery_id.nunique()
        marks = []
        for _, g in lr.groupby(lr.rest_before_h.round(0)):       # each long rest shared by the batteries
            if g.battery_id.nunique() >= 3:
                marks.append((int(g.cycle_valid.median()), g.rest_before_h.median()))
        for cyc, rest in sorted(marks):
            ytop = sub1[sub1.cycle_valid == cyc].soh.max()
            axes[0].annotate(f"rest {rest:.0f} h\n(~{rest / 24:.0f} days)", xy=(cyc, ytop), xytext=(cyc - 4, 0.985),
                             fontsize=8.5, ha="center", color=self.cfg.ink,
                             arrowprops=dict(arrowstyle="->", color=self.cfg.ink))
        if marks:
            axes[0].text(0.03, 0.04, f"Each upward spike = SOH jumps right after a long rest,\nseen in all {n_pin} batteries (capacity recovery).",
                         transform=axes[0].transAxes, fontsize=8.5, color=self.cfg.ink, va="bottom")
        # Noisy battery B0054 in the 2 A panel (4°C)
        if len(levels) > 1 and "B0054" in set(c.battery_id):
            g54 = c[c.battery_id == "B0054"]
            low = g54.loc[g54.soh.idxmin()]
            axes[1].annotate("B0054: noisy data\n(tier N), SOH jumps erratically", xy=(low.cycle_valid, low.soh),
                             xytext=(low.cycle_valid + 25, 0.60), fontsize=8.5, color=self.cfg.ink,
                             arrowprops=dict(arrowstyle="->", color=self.cfg.ink))
        handles = [plt.Line2D([], [], color=self.cfg.temp_colors[t], lw=2, label=f"ambient {t}") for t in self.cfg.temp_order]
        handles.append(plt.Line2D([], [], color=self.cfg.ink, lw=1.2, ls="--", label="square-wave discharge"))
        handles.append(plt.Line2D([], [], color="black", marker="x", ls="", label="reaches EOL (SOH 0.8)"))
        axes[-1].legend(handles=handles, fontsize=9, loc="lower left", frameon=False)
        fig.suptitle(f"SOH of each battery falls with the number of cycles (one line = one battery, {c.battery_id.nunique()} batteries)", fontsize=13)
        return self._save(fig, "desc_soh_curves.png")

    def impedance_scatter(self, group: str = "G1"):
        """Figure 6: Re and Rct against SOH in group G1 (4 batteries run to EOL).

        Colour + marker by battery, because each battery starts at a different impedance level:
        the pooled r mixes differences BETWEEN batteries with the ageing trend WITHIN each battery.
        Colours: first 3 colours of the standard palette + the secondary ink for the 4th battery; different
        markers so batteries are not told apart by colour alone.
        """
        g = self.data.cycles[self.data.cycles.group == group].dropna(subset=["Re_ohm", "Rct_ohm"])
        colors = ["#2a78d6", "#eb6834", "#1baf7a", self.cfg.ink]
        markers = ["o", "s", "^", "D"]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharex=True)
        for ax, var, label in zip(axes, ["Re_ohm", "Rct_ohm"], ["Re, electrolyte resistance (Ω)", "Rct, charge-transfer resistance (Ω)"]):
            within = []
            for (bid, gb), col, mk in zip(g.groupby("battery_id"), colors, markers):
                r_b = gb.soh.corr(gb[var])
                within.append(r_b)
                ax.scatter(gb.soh, gb[var], s=18, color=col, marker=mk, alpha=0.75, edgecolor="none",
                           label=f"{bid} (r = {r_b:.2f})")
            r_all = g.soh.corr(g[var])
            ax.set_title(f"{label.split(',')[0]} against SOH, group {group} (n = {len(g)} measurements)\n"
                         f"pooled r = {r_all:.2f}; r within each battery: {min(within):.2f} to {max(within):.2f}",
                         fontsize=10.5)
            ax.set_xlabel("SOH      (← older battery                    newer battery →)")
            ax.set_ylabel(label)
            self._clean_axes(ax)
        axes[0].legend(fontsize=8.5, frameon=False, loc="upper right", title="r within each battery", title_fontsize=8.5)
        axes[1].legend(fontsize=8.5, frameon=False, loc="lower left", title="r within each battery", title_fontsize=8.5)
        guide = ("How to read:\n• each dot = 1 impedance measurement (EIS, about every 4–5 cycles)\n"
                 "• HIGHER impedance = the battery resists current more\n• from right to left (battery ages): dots go UP\n"
                 "• one colour + marker per battery, because each battery\n   starts at a different impedance level")
        axes[0].text(0.02, 0.03, guide, transform=axes[0].transAxes, fontsize=8.3, va="bottom", ha="left", linespacing=1.45,
                     bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.8"))
        # Exception: B0018, Rct does not follow SOH
        if "B0018" in set(g.battery_id):
            gb = g[g.battery_id == "B0018"]
            hi = gb[gb.soh > 0.95].Rct_ohm
            r18 = gb.soh.corr(gb.Rct_ohm)
            if len(hi):
                axes[1].annotate(f"B0018: Rct barely follows SOH (r = {r18:.2f})"
                                 "\nhigh SOH but still high Rct, unlike the other 3 batteries",
                                 xy=(gb[gb.soh > 0.95].soh.mean(), hi.mean()), xytext=(0.60, 0.1145), fontsize=8.5, color=self.cfg.ink,
                                 arrowprops=dict(arrowstyle="->", color=self.cfg.ink))
                axes[1].set_ylim(top=0.118)
        fig.suptitle("The older the battery, the higher its impedance (group G1, 4 batteries run to EOL)", fontsize=13)
        return self._save(fig, "desc_impedance_soh.png")

    def all(self):
        return [self.soh_histogram(), self.rest_histogram(), self.tmax_boxplot(),
                self.fade_by_condition(), self.soh_curves(), self.impedance_scatter()]


# ----------------------------------------------------------------------------------------
# 5. Insights
# ----------------------------------------------------------------------------------------
class InsightWriter:
    """Writes insights following slide Chap 1, p. 60: FINDING -> NUMBER -> IMPLICATION.

    Every number in a sentence is computed from the data inside the method (README 9.7: no hand-typed numbers).
    Comparisons between conditions are HYPOTHESES only, to be tested in stage 3 (inferential statistics).
    """

    def __init__(self, data: DescriptiveData, stats: SummaryStats):
        self.data = data
        self.stats = stats

    @staticmethod
    def f(x, d=3):
        """Number with d decimals."""
        return f"{x:.{d}f}"

    @staticmethod
    def fi(x):
        """Integer with a thousands separator: 2065 -> 2,065."""
        return f"{int(x):,}"

    def soh_distribution(self) -> str:
        s = self.stats.describe(self.data.cycles.soh)
        return (f"**SOH is left-skewed.** At the cycle level (n = {self.fi(s['n'])}), SOH has mean {self.f(s['mean'])} "
                f"below the median {self.f(s['median'])}, skewness {self.f(s['skewness'], 2)}, kurtosis {self.f(s['kurtosis'], 2)} "
                f"(close to 3, the normal value). Most cycles are recorded while SOH is still high: Q1 = {self.f(s['Q1'])}, "
                f"so three quarters of the cycles have an SOH above this level. Implication: there are fewer data near the end "
                "of life, so models that forecast close to EOL have few examples to learn from; for a fleet operator this "
                "means end-of-life predictions carry the most uncertainty exactly when replacement decisions are made.")

    def rest_distribution(self) -> str:
        s = self.stats.describe(self.data.cycles.rest_before_h)
        n_long = int((self.data.cycles.rest_before_h > self.data.cfg.long_rest_h).sum())
        return (f"**Rest time is extremely right-skewed.** The median is only {self.f(s['median'] * 60, 1)} minutes but the mean is "
                f"{self.f(s['mean'], 2)} hours and the maximum {self.f(s['max'], 1)} hours; skewness {self.f(s['skewness'], 1)}, "
                f"kurtosis {self.f(s['kurtosis'], 0)}. Only {n_long} / {self.fi(s['n'])} cycles ({self.f(100 * n_long / s['n'], 1)}%) "
                "rest longer than 1 hour. Implication: the mean must not be used to describe this variable; in a model it must be "
                "log-transformed (the SVR uses log1p) or binned (the long_rest flag), following Chap 1, p. 17.")

    def recovery_after_rest(self) -> str:
        c = self.data.cycles.sort_values(["battery_id", "cycle_valid"])
        d_soh = c.groupby("battery_id").soh.diff()
        long_ = c.rest_before_h > self.data.cfg.long_rest_h
        up_share = (d_soh[long_] > 0).mean()
        med_long, med_short = d_soh[long_].median(), d_soh[~long_].median()
        n_bat = c[long_ & (d_soh > 0)].battery_id.nunique()
        return (f"**SOH jumps up after a long rest (capacity recovery).** After a rest of more than 1 hour, "
                f"{self.f(100 * up_share, 1)}% of cycles show a higher SOH than the cycle before (median change "
                f"{med_long:+.4f}), against a median change of {med_short:+.4f} after a short rest; the effect is seen in "
                f"{n_bat} batteries (tiers A/B/N). The gain is temporary: SOH falls back to the trend within a few cycles. Implication: "
                "these points are real behaviour, not errors, and must be kept; for warranty or resale checks, a single "
                "capacity test right after a vehicle has been parked for a long time overstates the battery's health.")

    def tmax_by_temp(self) -> str:
        t = self.stats.by_group(self.data.cycles, "T_max_C", "temp", self.data.cfg.temp_order)
        amb = self.data.cycles.groupby("temp").ambient_C.median()
        parts = [f"{k}: median {self.f(t.loc[k, 'median'], 1)}°C ({self.f(t.loc[k, 'median'] - amb[k], 1)}°C above ambient)"
                 for k in t.index]
        all_k = self.stats.describe(self.data.cycles.T_max_C)['kurtosis']
        hot_med = t.loc[self.data.cfg.temp_order[-1], "median"]
        return (f"**Cells heat up during discharge, and how much depends on the ambient.** {'; '.join(parts)}. "
                f"Pooled, T_max_C has kurtosis {self.f(all_k, 2)} < 3 because it mixes 3 groups, so it must be described "
                f"per ambient temperature. Implication: at 43°C ambient the cell reaches a median of {self.f(hot_med, 1)}°C during "
                "discharge; a cooling system for the pack is a hard requirement for vehicles used in hot climates.")

    def fade_by_condition(self) -> str:
        b = self.data.batteries[self.data.batteries.tier != "N"]
        m = b.groupby("condition")[["fade_pct_per_100cyc", "fade25_pct_per_100cyc"]].mean()
        n = b.groupby("condition").size()
        hot, warm = "43°C, 4 A, CC", "24°C, 4 A, CC"
        c1, c2 = "4°C, 1 A, CC", "4°C, 2 A, CC"
        return (f"**Degradation rates differ clearly between conditions (hypothesis, to be tested in stage 3).** "
                f"At the same 4 A, over the first 25 cycles, batteries at 43°C lose on average {self.f(m.loc[hot, 'fade25_pct_per_100cyc'], 1)} "
                f"SOH points per 100 cycles (n = {n[hot]}), against {self.f(m.loc[warm, 'fade25_pct_per_100cyc'], 1)} at 24°C (n = {n[warm]}). "
                f"At 4°C, 1 A degrades faster than 2 A: {self.f(m.loc[c1, 'fade_pct_per_100cyc'], 1)} against "
                f"{self.f(m.loc[c2, 'fade_pct_per_100cyc'], 1)} over the whole series, against intuition. "
                "The ranking of the groups changes with the measure (whole series or first 25 cycles), because degradation is "
                "not linear and the groups ran different numbers of cycles; the report must show both. Each group has only 1–5 "
                "batteries, so this is a description, not a conclusion. Implication (to be confirmed in stage 3): in this data, "
                "temperature looks like a stronger lever on battery life than discharge current, so thermal management deserves priority.")

    def impedance(self, group: str = "G1") -> str:
        g = self.data.cycles[self.data.cycles.group == group]
        r_re, r_rct = g.soh.corr(g.Re_ohm), g.soh.corr(g.Rct_ohm)
        w_re = g.groupby("battery_id").apply(lambda b: b.soh.corr(b.Re_ohm), include_groups=False)
        w_rct = g.groupby("battery_id").apply(lambda b: b.soh.corr(b.Rct_ohm), include_groups=False)
        return (f"**Impedance rises as the battery ages.** In {group} ({g.battery_id.nunique()} batteries), the pooled Pearson "
                f"correlation between SOH and Re is {self.f(r_re, 2)}, and between SOH and Rct {self.f(r_rct, 2)}. Within each battery, r with Re "
                f"ranges from {self.f(w_re.min(), 2)} to {self.f(w_re.max(), 2)}, and with Rct from {self.f(w_rct.min(), 2)} to {self.f(w_rct.max(), 2)}; "
                f"the exception is {w_rct.idxmax()} (r with Rct only {self.f(w_rct.max(), 2)}, Rct barely follows SOH). "
                "In the figure the points form one cluster per battery: each battery starts at a different impedance level, so "
                "the pooled r mixes differences between batteries with the ageing trend within each battery. Implication: impedance "
                "reflects ageing and supports the SEM model (operating conditions → impedance → SOH), but a battery's impedance should "
                "be compared with its own starting value; a fixed impedance threshold for all packs would misjudge some of them. "
                "Cycles within a battery are dependent, so r here is descriptive only, without p-values.")

    def c0_variation(self) -> str:
        s = self.data.batteries
        g1 = s[s.group == "G1"].C0_Ah
        below = int((g1 < 2.0).sum())
        low = s.loc[s.C0_Ah.idxmin()]
        mates = s[(s.group == low.group) & (s.battery_id != low.battery_id)].C0_Ah    # batteries in the same group as the lowest one
        return (f"**Batteries differ from the start.** C0 of the {len(g1)} G1 batteries (same conditions) ranges from {self.f(g1.min())} "
                f"to {self.f(g1.max())} Ah, a spread of {self.f(100 * (g1.max() - g1.min()) / g1.min(), 1)}%; {below} / {len(g1)} batteries "
                f"are below the rated 2.0 Ah. {low.battery_id} has a C0 of only {self.f(low.C0_Ah)} Ah, while the other batteries of group {low.group} "
                f"have {self.f(mates.min())}–{self.f(mates.max())} Ah, "
                "most likely a defective cell from the start. Implication: SOH must be normalised by each battery's own C0, not by 2.0 Ah; "
                "for warranty, the spread between cells is a risk of its own and incoming cells should be screened.")

    def eol_censoring(self) -> str:
        b = self.data.batteries
        reached = int(b.cycles_to_soh80.notna().sum())
        short = b[b.cycles_to_soh80.isna() & (b.tier == "B")]
        stops = short.groupby("condition").n_used.agg(["min", "max", "size"])
        stops = stops[stops["size"] >= 2]                     # groups whose batteries stopped together
        parts = [f"{k}: {int(r['size'])} batteries stopped after {int(r['min'])}" + (f"–{int(r['max'])}" if r['max'] != r['min'] else "")
                 + " cycles" for k, r in stops.iterrows()]
        return (f"**Almost half of the batteries never reach EOL in the data.** Only {reached} / {len(b)} batteries fall below SOH 0.8 "
                f"(column cycles_to_soh80); the other {len(b) - reached} stop earlier. Several groups stop together after the same "
                f"number of cycles ({'; '.join(parts)}), which points to the experiment being ended, not to the batteries failing; "
                "the dataset does not record why. Implication: the mean of cycles_to_soh80 must not be read as the lifetime of all "
                "batteries (the data are censored), and a condition with few batteries at EOL is not a 'gentler' condition.")

    def all(self) -> list:
        return [self.soh_distribution(), self.rest_distribution(), self.recovery_after_rest(), self.tmax_by_temp(),
                self.fade_by_condition(), self.impedance(), self.c0_variation(), self.eol_censoring()]


# ----------------------------------------------------------------------------------------
# 6. Full run
# ----------------------------------------------------------------------------------------
class DescriptiveAnalysis:
    def __init__(self, config: DescConfig = None):
        self.cfg = config or DescConfig()
        self.data = DescriptiveData(self.cfg).load()
        self.stats = SummaryStats()
        self.plots = DescriptivePlots(self.cfg, self.data)
        self.insights = InsightWriter(self.data, self.stats)
        self.scales = MeasurementScales()

    def scale_table(self) -> pd.DataFrame:
        """C1.1: scale of measurement of every column, with the reason."""
        return self.scales.table(self.data)

    def cycle_table(self) -> pd.DataFrame:
        return self.stats.table(self.data.cycles, self.cfg.cycle_vars)

    def battery_table(self) -> pd.DataFrame:
        return self.stats.table(self.data.batteries, self.cfg.battery_vars)

    def cycle_tables_by_condition(self) -> dict:
        """C1.5: cycle-level measures of each variable in cfg.condition_vars, one row per operating condition."""
        order = self.data.condition_order()
        return {v: self.stats.by_group(self.data.cycles, v, "condition", order) for v in self.cfg.condition_vars}

    def battery_table_by_condition(self) -> pd.DataFrame:
        """C1.5: battery-level C0 and final SOH by operating condition (without the noisy tier N battery)."""
        b = self.data.batteries[self.data.batteries.tier != "N"]
        order = self.data.condition_order()
        rows = {v: self.stats.by_group(b, v, "condition", order)[["n", "mean", "SD", "min", "max"]]
                for v in ["C0_Ah", "soh_last"]}
        return pd.concat(rows, axis=1)

    def fade_table(self) -> pd.DataFrame:
        """Degradation rate by operating condition, without the noisy battery (tier N)."""
        b = self.data.batteries[self.data.batteries.tier != "N"]
        order = self.data.condition_order()
        rows = {}
        for var in ["fade_pct_per_100cyc", "fade25_pct_per_100cyc"]:
            t = self.stats.by_group(b, var, "condition", order)[["n", "mean", "SD", "min", "max"]]
            rows[var] = t
        return pd.concat(rows, axis=1)

    def run(self):
        self.plots.all()
        return self


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")      # Windows consoles: allow °C, → and other symbols
    pd.set_option("display.width", 160)
    a = DescriptiveAnalysis().run()
    print(a.cycle_table().round(3))
    print(a.battery_table().round(3))
    print(a.fade_table().round(1))
    for s in a.insights.all():
        print("-", s)
