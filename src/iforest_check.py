"""Isolation Forest cross-check of the outlier flags (Minh Vu, branch minh-vu).

Go.md stage 1 step 1.5, WBS C6.1, slides Chap 6 §5 (pp. 69–79).
NOT a modelling algorithm (Rec.md section 2, row 12): only used to compare the `cap_outlier` flag
(made with a Hampel filter or the > 25% drop rule) with an independent ML method.
The results do not change the data or the agreed flags.

Classes:
    IForestConfig       Settings: tiers, features, seed, score threshold, colours.
    IForestData         Reads cycles_clean.csv (tiers A, B), adds ΔSOH and the position of each cycle in the battery's life.
    IsolationForestChecker
                        Fits one iForest per experiment group, computes the anomaly score s(x, n),
                        compares with cap_outlier, sorts the points flagged by iForest only,
                        checks stability across seeds and compares at the same flagging rate.
    IForestPlots        2 figures: ΔSOH against rest time, and SOH curves of a few example batteries.
    IForestInsights     Insight sentences; every number comes from the results.
    IForestAnalysis     Calls the classes above in order.

How Isolation Forest works (pp. 75–78):
    Each tree splits the data with random cuts (a random feature and a random threshold)
    until every point is on its own. Anomalies are isolated after FEW cuts.
    Anomaly score:  s(x, n) = 2 ^ ( -E[h(x)] / c(n) )
        h(x): number of cuts needed to isolate x;  E[h(x)]: mean over the trees;
        c(n): average path length to isolate one point among n points (normalisation).
    s close to 1: very anomalous;  s well below 0.5: normal.
    sklearn: score_samples() returns -s(x, n); contamination='auto' flags a point when s > 0.5.

Run from the repository root:
    python src/iforest_check.py
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

# Category names (one per row)
BOTH, HAMPEL_ONLY, IF_REST, IF_OTHER, NORMAL = (
    "both", "Hampel only", "iForest only: after long rest", "iForest only: other", "normal")
LONG, SHORT = "rest > 1 h", "rest <= 1 h"


# ----------------------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------------------
@dataclass
class IForestConfig:
    tiers: tuple = ("A", "B")                     # Go.md 1.5: tiers A and B only
    # Features from Go.md 1.5. d_soh = SOH of this cycle - SOH of the previous valid cycle (same battery)
    features: tuple = ("soh", "d_soh", "T_max_C", "V_min", "rest_before_h")
    seed: int = 42
    stability_seeds: tuple = (0, 1, 2, 3, 4)      # other seeds, to see whether the result is stable
    n_estimators: int = 100                       # number of trees (sklearn default)
    score_threshold: float = 0.5                  # contamination='auto' <=> flag when s > 0.5
    edge_frac: float = 0.10                       # "start/end of life" = first or last 10% of a battery's cycles

    # Colours: first 3 colours of the standard palette + dark grey + light grey for normal points
    colors: dict = field(default_factory=lambda: {
        BOTH: "#2a78d6",
        HAMPEL_ONLY: "#eb6834",
        IF_REST: "#1baf7a",
        IF_OTHER: "#52514e",
        NORMAL: "#d4d4d4",
    })
    markers: dict = field(default_factory=lambda: {
        BOTH: "o", HAMPEL_ONLY: "s", IF_REST: "^", IF_OTHER: "D", NORMAL: ".",
    })
    example_batteries: tuple = ("B0005", "B0033", "B0055")
    fig_dir: Path = ROOT / "reports" / "figures"


# ----------------------------------------------------------------------------------------
# 2. Data
# ----------------------------------------------------------------------------------------
class IForestData:
    """Data for iForest.

    DIFFERENT from the descriptive statistics: the cap_outlier and break_in rows are KEPT here,
    because the point is to check those flags.
    """

    def __init__(self, config: IForestConfig):
        self.cfg = config

    def load(self) -> pd.DataFrame:
        clean = pd.read_csv(DATA / "cycles_clean.csv")
        d = clean[clean.tier.isin(self.cfg.tiers)].sort_values(["battery_id", "cycle_valid"]).reset_index(drop=True)
        # ΔSOH against the previous valid cycle of the same battery; the first cycle has none -> 0
        d["d_soh"] = d.groupby("battery_id").soh.diff().fillna(0.0)
        # Position of the cycle in the battery's life (0..1), to spot cycles at the start or end of a series
        d["life_pos"] = d.groupby("battery_id").cycle_valid.rank(pct=True)
        self.df = d
        return d


# ----------------------------------------------------------------------------------------
# 3. Isolation Forest and comparison
# ----------------------------------------------------------------------------------------
class IsolationForestChecker:
    """Fits one iForest PER experiment group (G1, G2, ...).

    Why per group: each group has different operating conditions (temperature, current), so the
    T_max_C or V_min of the 43°C group would look "anomalous" next to the 4°C group, although it is
    perfectly normal within its own group.
    No scaling needed: each cut looks at one feature only, so units do not matter.
    """

    def __init__(self, config: IForestConfig, df: pd.DataFrame):
        self.cfg = config
        self.df = df

    def fit_score(self, seed: int, contamination="auto"):
        """Returns (s, flag): anomaly score s(x, n) and the iForest flag for every row."""
        F = list(self.cfg.features)
        s = pd.Series(np.nan, index=self.df.index)
        flag = pd.Series(False, index=self.df.index)
        for _, g in self.df.groupby("group"):
            m = IsolationForest(n_estimators=self.cfg.n_estimators, contamination=contamination,
                                random_state=seed).fit(g[F])
            s[g.index] = -m.score_samples(g[F])            # flip the sign to get s(x, n) as on the slides
            flag[g.index] = m.predict(g[F]) == -1
        return s, flag

    def run(self) -> pd.DataFrame:
        s, flag = self.fit_score(self.cfg.seed)
        # Check: with contamination='auto', sklearn's flag is exactly "s > 0.5"
        assert (flag == (s > self.cfg.score_threshold)).all()
        d = self.df
        d["s_score"] = s
        d["if_flag"] = flag
        d["category"] = self.categorize(d)
        return d

    def categorize(self, d: pd.DataFrame) -> pd.Series:
        """Each row goes into exactly 1 of 5 categories. Points flagged by iForest only are split by reason."""
        both = d.cap_outlier & d.if_flag
        hampel_only = d.cap_outlier & ~d.if_flag
        if_only = ~d.cap_outlier & d.if_flag
        return pd.Series(np.select(
            [both, hampel_only, if_only & d.long_rest, if_only],
            [BOTH, HAMPEL_ONLY, IF_REST, IF_OTHER],
            NORMAL), index=d.index)

    # ---- Tables ------------------------------------------------------------------------
    def agreement_table(self) -> pd.DataFrame:
        """Cross table cap_outlier × iForest flag (with totals)."""
        t = pd.crosstab(self.df.cap_outlier, self.df.if_flag, margins=True, margins_name="Total")
        t.index = t.index.map({False: "Hampel: no", True: "Hampel: yes", "Total": "Total"})
        t.columns = t.columns.map({False: "iForest: no", True: "iForest: yes", "Total": "Total"})
        return t

    def category_table(self) -> pd.DataFrame:
        order = list(self.cfg.colors)
        n = self.df.category.value_counts().reindex(order).fillna(0).astype(int)
        return pd.DataFrame({"rows": n, "% of all": (100 * n / len(self.df)).round(2)})

    def by_group(self) -> pd.DataFrame:
        t = pd.crosstab(self.df.group, self.df.category).reindex(columns=list(self.cfg.colors), fill_value=0)
        t["rows"] = self.df.groupby("group").size()
        return t

    def long_rest_table(self) -> pd.DataFrame:
        """Cycles after a long rest (> 1 hour): what ΔSOH does, and how often each method flags them."""
        rows = []
        for name, mask in [(LONG, self.df.long_rest), (SHORT, ~self.df.long_rest)]:
            g = self.df[mask]
            rows.append({"rest": name, "rows": len(g),
                         "median ΔSOH": g.d_soh.median(), "% with ΔSOH > 0": 100 * (g.d_soh > 0).mean(),
                         "% flagged by iForest": 100 * g.if_flag.mean(),
                         "% flagged by Hampel": 100 * g.cap_outlier.mean(),
                         "median s(x, n)": g.s_score.median()})
        return pd.DataFrame(rows).set_index("rest")

    def if_only_reasons(self) -> pd.DataFrame:
        """Points flagged by iForest only, split by likely reason."""
        g = self.df[~self.df.cap_outlier & self.df.if_flag]
        edge = (g.life_pos <= self.cfg.edge_frac) | (g.life_pos >= 1 - self.cfg.edge_frac)
        reason = np.select([g.long_rest, edge], ["after a long rest (long_rest)", "first or last 10% of life"],
                           "other")
        n = pd.Series(reason).value_counts()
        return pd.DataFrame({"rows": n, "% of iForest-only points": (100 * n / len(g)).round(1)})

    def seed_stability(self) -> pd.DataFrame:
        """Reruns with other seeds: number of flagged points and overlap (Jaccard) with the seed-42 result."""
        base = self.df.if_flag
        n_h = int(self.df.cap_outlier.sum())
        caught = f"Hampel flags caught (of {n_h})"
        rows = [{"seed": self.cfg.seed, "points flagged": int(base.sum()), "Jaccard with seed 42": 1.0,
                 caught: int((base & self.df.cap_outlier).sum())}]
        for sd in self.cfg.stability_seeds:
            _, f = self.fit_score(sd)
            rows.append({"seed": sd, "points flagged": int(f.sum()),
                         "Jaccard with seed 42": (f & base).sum() / (f | base).sum(),
                         caught: int((f & self.df.cap_outlier).sum())})
        return pd.DataFrame(rows).set_index("seed")

    def same_rate_comparison(self) -> pd.DataFrame:
        """Fair comparison: let iForest flag EXACTLY Hampel's share of rows, then count the overlap."""
        rate = self.df.cap_outlier.mean()
        _, f = self.fit_score(self.cfg.seed, contamination=rate)
        self.same_rate_ = rate
        d = self.df
        return pd.DataFrame([{
            "flag rate": rate, "iForest flags": int(f.sum()), "Hampel flags": int(d.cap_outlier.sum()),
            "both": int((f & d.cap_outlier).sum()),
            "iForest only": int((f & ~d.cap_outlier).sum()),
            "of which after a long rest": int((f & ~d.cap_outlier & d.long_rest).sum()),
        }])


# ----------------------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------------------
class IForestPlots:
    def __init__(self, config: IForestConfig, df: pd.DataFrame):
        self.cfg = config
        self.df = df

    @staticmethod
    def _clean(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="0.92", lw=0.8)
        ax.set_axisbelow(True)

    def _scatter_by_category(self, ax, d, x, y, size=26):
        # normal points first (underneath), flagged points last (on top)
        for cat in [NORMAL, IF_OTHER, IF_REST, HAMPEL_ONLY, BOTH]:
            g = d[d.category == cat]
            if len(g):
                ax.scatter(g[x], g[y], s=8 if cat == NORMAL else size, color=self.cfg.colors[cat],
                           marker=self.cfg.markers[cat], edgecolor="white" if cat != NORMAL else "none",
                           linewidth=0.5, alpha=0.9, label=f"{cat} ({len(g)})", zorder=2 if cat == NORMAL else 3)

    def rest_scatter(self):
        """Figure 1: ΔSOH against rest time (log scale). Recovery points after long rests sit at the top right."""
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        self._scatter_by_category(ax, self.df, "rest_before_h", "d_soh")
        ax.set_xscale("log")
        ax.axvline(1.0, color="0.45", ls="--", lw=1)
        ax.axhline(0, color="0.45", lw=0.8)
        ax.text(1.05, ax.get_ylim()[1] * 0.92, "rest > 1 h (long_rest)", color="0.35", fontsize=9)
        ax.set_xlabel("rest_before_h: rest before the discharge (hours, log scale)")
        ax.set_ylabel("ΔSOH against the previous cycle")
        ax.set_title(f"Which points do Hampel and Isolation Forest flag? (tiers A/B, n = {len(self.df):,})")
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
        self._clean(ax)
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "iforest_rest_scatter.png", dpi=150)
        return fig

    def example_curves(self):
        """Figure 2: SOH curves of a few example batteries, with the flagged points marked."""
        ids = list(self.cfg.example_batteries)
        fig, axes = plt.subplots(len(ids), 1, figsize=(12.5, 3.1 * len(ids)), sharey=False)
        for ax, bid in zip(np.atleast_1d(axes), ids):
            b = self.df[self.df.battery_id == bid]
            ax.plot(b.cycle_valid, b.soh, color="0.6", lw=1, zorder=1)
            self._scatter_by_category(ax, b[b.category != NORMAL], "cycle_valid", "soh", size=40)
            lr = b[b.long_rest]
            ax.scatter(lr.cycle_valid, lr.soh, s=110, facecolor="none", edgecolor="0.35", linewidth=0.8, zorder=2,
                       label="grey ring = cycle after a long rest")
            c = b.iloc[0]
            ax.set_title(f"{bid} ({c.group}, {c.ambient_C:.0f}°C, {c.I_level_A:.0f} A)", fontsize=10.5)
            ax.set_ylabel("SOH")
            ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))   # legend outside the panel
            self._clean(ax)
        np.atleast_1d(axes)[-1].set_xlabel("cycle_valid")
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "iforest_examples.png", dpi=150)
        return fig


# ----------------------------------------------------------------------------------------
# 5. Insights
# ----------------------------------------------------------------------------------------
class IForestInsights:
    """Insight sentences; every number is computed from the IsolationForestChecker results."""

    def __init__(self, checker: IsolationForestChecker):
        self.ck = checker
        self.d = checker.df
        self.cfg = checker.cfg

    @staticmethod
    def f(x, d=1):
        return f"{x:.{d}f}"

    def agreement(self) -> str:
        d = self.d
        n_h, n_i = int(d.cap_outlier.sum()), int(d.if_flag.sum())
        both = int((d.cap_outlier & d.if_flag).sum())
        return (f"**iForest agrees with almost every Hampel flag, but flags far more points.** "
                f"Of the {n_h} rows flagged by Hampel, iForest also flags {both} ({self.f(100 * both / n_h)}%). "
                f"In total iForest flags {n_i} rows ({self.f(100 * n_i / len(d))}% of the data), "
                f"{self.f(n_i / n_h)} times as many as Hampel. So the `cap_outlier` flags are confirmed by an independent method; "
                "the remaining question is whether the extra iForest points are wrong data.")

    def recovery(self) -> str:
        t = self.ck.long_rest_table()
        a, b = t.loc[LONG], t.loc[SHORT]
        r = self.ck.if_only_reasons()
        n_rest = int(r.loc["after a long rest (long_rest)", "rows"])
        n_if_only = int(r["rows"].sum())
        return (f"**Almost half of the extra iForest points are capacity recoveries after a rest.** "
                f"After a rest of more than 1 hour, the median ΔSOH is {a['median ΔSOH']:+.4f} and {self.f(a['% with ΔSOH > 0'])}% of cycles "
                f"show a higher SOH, against a median of {b['median ΔSOH']:+.4f} after a short rest. iForest flags "
                f"{self.f(a['% flagged by iForest'])}% of the long-rest cycles, Hampel only {self.f(a['% flagged by Hampel'])}%. "
                f"Of the {n_if_only} points flagged by iForest only, {n_rest} ({self.f(100 * n_rest / n_if_only)}%) come right after a long rest.")

    def why_not_outlier(self) -> str:
        lr = self.d[self.d.long_rest]
        n_bat = lr[lr.d_soh > 0].battery_id.nunique()
        n_grp = lr[lr.d_soh > 0].group.nunique()
        return ("**Why the recovery points should NOT be treated as outliers.** iForest flags them because they are rare and different: "
                "SOH jumps up against the downward trend, together with an unusually long rest. But this is a real electrochemical "
                "effect (capacity recovery, F5 in Introduction_of_dataset), usually explained as follows: while the battery rests, "
                "the lithium-ion concentration inside the electrodes has time to even out, so the capacity rises for a while and then "
                "falls back to the trend after a few cycles (see the example figure). It happens exactly when there is a cause (a long rest), "
                f"it repeats in {n_bat} batteries from {n_grp} experiment groups, "
                "and it has a fixed direction (SOH goes up), unlike measurement errors, which are random. Deleting these points would lose "
                "information: rest time is one of the most important non-history features of the SVR model. So the team keeps these points and "
                "uses the `long_rest` flag as an explanatory variable. Lesson: \"statistically unusual\" does not mean \"wrong data\"; "
                "a decision to remove points must rest on an understanding of the phenomenon.")

    def same_rate(self) -> str:
        t = self.ck.same_rate_comparison().iloc[0]
        return (f"**Comparison at the same flagging rate.** If iForest may only flag {self.f(100 * t['flag rate'], 2)}% of the data like Hampel "
                f"({int(t['iForest flags'])} rows), the two methods share only {int(t['both'])} rows; "
                f"{int(t['of which after a long rest'])} / {int(t['iForest only'])} of the remaining points come after a long rest. "
                "So the points iForest finds \"most unusual\" are mainly recoveries, not only the deviations from the trend that Hampel "
                "targets. The two methods answer different questions: Hampel looks for points that leave the trend of their own series; "
                "iForest looks for rare points in a space of several features.")

    def hampel_only(self) -> str:
        h = self.d[self.d.category == HAMPEL_ONLY]
        if not len(h):
            return "**No point is flagged by Hampel only.**"
        items = ", ".join(f"{r.battery_id} cycle {int(r.cycle_valid)} (ΔSOH {r.d_soh:.3f}, s = {r.s_score:.2f})"
                          for r in h.itertuples())
        groups = ", ".join(sorted(h.group.unique()))
        return (f"**{len(h)} points flagged by Hampel only:** {items}. Their score s is close to the 0.5 threshold: they leave the trend "
                f"of their own battery (what Hampel looks at), but are not rare enough within the whole group {groups}, where SOH already varies a lot.")

    def other_if_only(self) -> str:
        g = self.d[(self.d.category == IF_OTHER)]
        edge = (g.life_pos <= self.cfg.edge_frac) | (g.life_pos >= 1 - self.cfg.edge_frac)
        # for the remaining points: which feature is furthest from its group (measured as |x - median| / IQR)
        F = list(self.cfg.features)
        z = self.d.groupby("group")[F].transform(lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25) + 1e-12)).abs()
        top = z.loc[g[~edge].index].idxmax(axis=1).value_counts()
        top_s = ", ".join(f"{k} ({v})" for k, v in top.items())
        return (f"**Points flagged by iForest only, not after a long rest ({len(g)} points).** {int(edge.sum())} of them lie in the first "
                "or last 10% of the battery's life: at both ends of a series, SOH or ΔSOH tends to sit at the edge of the group's range and is "
                "isolated with few cuts, although this may just be normal behaviour (start-up or end of life). For the other "
                f"{int((~edge).sum())} points, the feature furthest from its group is: {top_s}; "
                "most are moderate ΔSOH jumps that stay below the Hampel threshold. This follows from contamination='auto' "
                "(the s > 0.5 threshold is fairly loose), not from wrong data.")

    def stability(self) -> str:
        t = self.ck.seed_stability()
        j = t.iloc[1:]["Jaccard with seed 42"]
        return (f"**Stability.** Rerun with {len(j)} other seeds, the number of flagged points ranges from {int(t['points flagged'].min())} "
                f"to {int(t['points flagged'].max())}, and the Jaccard overlap with seed 42 from {self.f(j.min(), 2)} to {self.f(j.max(), 2)}. "
                "Points close to the 0.5 threshold can switch between runs, but the conclusion about the recovery points does not change.")

    def all(self) -> list:
        return [self.agreement(), self.recovery(), self.why_not_outlier(), self.other_if_only(), self.same_rate(), self.hampel_only(), self.stability()]


# ----------------------------------------------------------------------------------------
# 6. Full run
# ----------------------------------------------------------------------------------------
class IForestAnalysis:
    def __init__(self, config: IForestConfig = None):
        self.cfg = config or IForestConfig()
        self.df = IForestData(self.cfg).load()
        self.checker = IsolationForestChecker(self.cfg, self.df)
        self.checker.run()
        self.plots = IForestPlots(self.cfg, self.df)
        self.insights = IForestInsights(self.checker)

    def run(self):
        self.plots.rest_scatter()
        self.plots.example_curves()
        return self


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")      # Windows consoles: allow Δ, °C and other symbols
    pd.set_option("display.width", 160)
    a = IForestAnalysis().run()
    ck = a.checker
    for t in (ck.agreement_table(), ck.category_table(), ck.by_group(), ck.long_rest_table().round(4),
              ck.if_only_reasons(), ck.seed_stability().round(3), ck.same_rate_comparison().round(4)):
        print(t, "\n")
    for s in a.insights.all():
        print("-", s)
