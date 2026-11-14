"""SVR model that forecasts next-cycle SOH (Minh Vu, branch minh-vu).

Answers RQ4: how well do charge and impedance features forecast next-cycle SOH,
and which features matter most. Plan: docs/Go.md stage 5c, WBS C6.2 to C6.5.

The code is split into classes, each with one job:

    SVRConfig            All settings in one place (number of lags, parameter grid, seed...).
    FeatureBuilder       Builds the features of each cycle and checks for data leakage.
    RecursiveForecaster  Multi-step recursive forecast (scenario E2).
    SVRTuner             Chooses C, epsilon, gamma with GroupKFold by battery + the one-SE rule.
    SVRForecaster        Fits the final model, forecasts E1 and E2, builds the shared prediction file.
    ImportanceAnalyzer   Permutation importance: which features matter (RQ4).
    SVRPlotter           Forecast, tuning and importance figures.
    SVRExperiment        Calls the classes above in order, from loading the data to saving files.

Flow (SVRExperiment.run):
    load_split() -> FeatureBuilder -> SVRTuner -> SVRForecaster -> evaluate() -> ImportanceAnalyzer -> SVRPlotter

Data, the train/test split and the metrics all come from src/protocol.py (shared by the team),
so the results are comparable with the other models.

Run from the repository root:
    python src/svr_model.py
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from protocol import EOL_SOH, KEYS, ORIGIN_FRACTION, PRED_COLS, evaluate, load_split, naive

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------------------
@dataclass
class SVRConfig:
    """Every hand-picked number of the model lives here, so it is easy to find and to report."""

    n_lags: int = 5                 # number of previous cycles used as features (soh_lag1..soh_lag5)
    n_splits: int = 4               # GroupKFold with k = 4, grouped by battery (docs/Rec.md section 6)
    seed: int = 42
    n_repeats: int = 30             # number of shuffles in permutation importance
    interval_level: float = 0.95    # level of the E1 prediction interval

    # Final grid = every value of Go.md + an extension.
    # Why extend: on the original grid the best value fell on the edge for all 3 parameters.
    param_grid: dict = field(default_factory=lambda: {
        "svr__C":       [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
        "svr__epsilon": [0.0001, 0.0003, 0.001, 0.005, 0.01],
        "svr__gamma":   ["scale", 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1],
    })
    # Original grid of docs/Go.md stage 5c, kept so we can report the result on this grid
    original_grid: dict = field(default_factory=lambda: {
        "svr__C":       [0.1, 1, 10, 100],
        "svr__epsilon": [0.001, 0.005, 0.01],
        "svr__gamma":   ["scale", 0.01, 0.1],
    })

    # Measured features (unknown after the E2 origin), held at their last known value
    held_features: tuple = ("chg_cc_time_s", "Re_ohm", "Rct_ohm", "log_rest_h")

    # Columns that must never be features: they leak the cycle being forecast (Go.md 4.6)
    forbidden: frozenset = frozenset({
        "soh", "capacity_Ah", "capacity_to_2p7V_Ah", "capacity_int_Ah", "energy_Wh", "duration_s",
        "t_to_cutoff_s", "T_max_C", "V_min", "C0_Ah", "soh_segment"})

    # Output locations
    fig_dir: Path = ROOT / "reports" / "figures"
    pred_path: Path = ROOT / "reports" / "predictions" / "svr.csv"


# ----------------------------------------------------------------------------------------
# 2. Features
# ----------------------------------------------------------------------------------------
class FeatureBuilder:
    """Turns the cycle table into the feature matrix X.

    Rule: for the discharge at cycle t, only use information known BEFORE that discharge.
    - soh_lag1..5, soh_roll5: SOH of the 5 previous cycles (same battery) and their mean
    - chg_cc_time_s: duration of the CC phase of the charge just before (measured before discharge)
    - Re_ohm, Rct_ohm: impedance from the last EIS BEFORE cycle t (joined during preprocessing)
    - T_max_prev_C: maximum temperature of the PREVIOUS discharge (t-1), not of t
    - log_rest_h: log(1 + rest time), because rest time is extremely right-skewed
    - is_4C, I_level_A, cutoff_V: operating conditions, fixed by the experiment design
    """

    def __init__(self, config: SVRConfig):
        self.cfg = config
        self.lag_cols = [f"soh_lag{k}" for k in range(1, config.n_lags + 1)]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns X with the same index as df. Lags are computed within each battery."""
        g = df.groupby("battery_id")
        X = pd.DataFrame(index=df.index)
        for k in range(1, self.cfg.n_lags + 1):
            X[f"soh_lag{k}"] = g.soh.shift(k)          # shift(k): value from k cycles earlier
        X["soh_roll5"] = X[self.lag_cols].mean(axis=1)
        X["chg_cc_time_s"] = df.chg_cc_time_s
        X["Re_ohm"] = df.Re_ohm
        X["Rct_ohm"] = df.Rct_ohm
        X["T_max_prev_C"] = g.T_max_C.shift(1)
        X["log_rest_h"] = np.log1p(df.rest_before_h)
        X["is_4C"] = (df.ambient_C < 10).astype(float)  # tier A only has 4°C and 24°C, so one dummy is enough
        X["I_level_A"] = df.I_level_A
        X["cutoff_V"] = df.discharge_cutoff_V
        return X

    @property
    def feature_names(self) -> list:
        return self.lag_cols + ["soh_roll5", "chg_cc_time_s", "Re_ohm", "Rct_ohm", "T_max_prev_C",
                                "log_rest_h", "is_4C", "I_level_A", "cutoff_V"]

    def check_no_leakage(self):
        """Stops the program if a forbidden column is in the feature list."""
        leaked = self.cfg.forbidden & set(self.feature_names)
        assert not leaked, f"forbidden columns used as features: {leaked}"

    def training_matrix(self, train: pd.DataFrame):
        """X, y, groups for the training set.

        The first 5 cycles of each battery have no full history (lags are NaN), so they are dropped.
        Rows with missing Re/Rct (no EIS measurement yet) are also dropped, NOT imputed (Rec.md 6.1 #6).
        groups = battery_id, used by GroupKFold so that each fold holds out whole batteries.
        """
        X = self.build(train)
        has_history = X[self.lag_cols + ["T_max_prev_C", "Re_ohm", "Rct_ohm"]].notna().all(axis=1)
        return X[has_history], train.soh[has_history], train.battery_id[has_history]


# ----------------------------------------------------------------------------------------
# 3. Recursive forecast (E2)
# ----------------------------------------------------------------------------------------
class RecursiveForecaster:
    """Stands at an origin and forecasts the rest of one battery's series.

    Each new forecast is pushed into the "history" and becomes soh_lag1 of the next step.
    Measurements unknown after the origin (cfg.held_features, T_max_prev_C) are held at their
    last value before the origin. Operating conditions (is_4C, I_level_A, cutoff_V) are fixed, so
    they are used as they are.
    This class is used in 2 places: tuning (SVRTuner) and the E2 forecast on the test set (SVRForecaster).
    """

    def __init__(self, builder: FeatureBuilder):
        self.builder = builder
        self.held = list(builder.cfg.held_features)

    def forecast(self, model, battery: pd.DataFrame, origin: int) -> np.ndarray:
        """Forecasts rows origin..end of ONE battery. Only uses information before row `origin`."""
        b = battery.reset_index(drop=True)
        Xb = self.builder.build(b)
        history = list(b.soh[:origin])                 # true SOH before the origin
        last_known = Xb.iloc[origin - 1]
        t_max_last = b.T_max_C.iloc[origin - 1]        # max temperature of the last known discharge
        out = []
        for i in range(origin, len(b)):
            x = Xb.iloc[[i]].copy()
            for k in range(1, self.builder.cfg.n_lags + 1):
                x[f"soh_lag{k}"] = history[-k]         # lags come from the history (true + forecast)
            x["soh_roll5"] = np.mean(history[-self.builder.cfg.n_lags:])
            x[self.held] = last_known[self.held].values
            x["T_max_prev_C"] = t_max_last
            y_hat = model.predict(x[self.builder.feature_names])[0]
            history.append(y_hat)
            out.append(y_hat)
        return np.array(out)


# ----------------------------------------------------------------------------------------
# 4. Hyperparameter selection
# ----------------------------------------------------------------------------------------
class SVRTuner:
    """Chooses C, epsilon and gamma using the training data only (the test set is never used).

    Step 1: GridSearchCV with GroupKFold(4) by battery, score = one-step-ahead RMSE.
    Step 2 (one-SE rule): many settings have almost the same one-step RMSE but behave very
            differently when forecasting recursively. Keep the settings whose RMSE is within one
            standard error of the best one, then pick the one with the lowest recursive (E2-style) RMSE.

    Meaning of the parameters (Chap 6, p. 31):
    - C: how hard errors larger than epsilon are penalised. Large C -> follows the training data closely.
    - epsilon: errors smaller than this count as zero (in SOH units).
    - gamma: how local the RBF kernel is. Small -> smooth, almost linear function; large -> wiggly function.
    Note: C and gamma trade off against each other. With a small gamma the model is nearly linear,
    so (C = 1, gamma = 3e-4), (C = 3, gamma = 1e-4), (C = 10, gamma = 3e-5) give almost the same fit.
    """

    def __init__(self, config: SVRConfig, builder: FeatureBuilder, forecaster: RecursiveForecaster):
        self.cfg = config
        self.builder = builder
        self.forecaster = forecaster
        self.cv = GroupKFold(n_splits=config.n_splits)

    @staticmethod
    def make_pipeline():
        """Scaler -> SVR in one Pipeline.

        In each fold, the mean/SD used for scaling are computed on the training part of that fold
        only (no leakage into the held-out part). There is no imputation step: incomplete rows are dropped.
        """
        return make_pipeline(StandardScaler(), SVR(kernel="rbf"))

    # ---- Step 1 -------------------------------------------------------------------------
    def grid_search(self, X, y, groups) -> pd.DataFrame:
        gs = GridSearchCV(self.make_pipeline(), self.cfg.param_grid, cv=self.cv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        self.grid_ = gs
        res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        res["cv_RMSE"] = -res.mean_test_score
        res["cv_RMSE_sd"] = res.std_test_score
        self.cv_results_ = res
        return res

    def best_on_original_grid(self) -> pd.Series:
        """Best setting if only the original Go.md grid were used (for the report)."""
        r, og = self.cv_results_, self.cfg.original_grid
        mask = (r.param_svr__C.isin(og["svr__C"]) & r.param_svr__epsilon.isin(og["svr__epsilon"])
                & r.param_svr__gamma.astype(str).isin([str(v) for v in og["svr__gamma"]]))
        return r[mask].iloc[0]

    def compare_with_naive(self, X, y, groups) -> pd.DataFrame:
        """Per-fold RMSE of the best setting (step 1) against naive SOH(t) = SOH(t-1)."""
        best_i = self.grid_.best_index_
        rows = []
        for i, (_, va) in enumerate(self.cv.split(X, y, groups)):
            rows.append({"fold": i, "held_out": ", ".join(sorted(groups.iloc[va].unique())),
                         "SVR_RMSE": -self.grid_.cv_results_[f"split{i}_test_score"][best_i],
                         "naive_RMSE": np.sqrt(((y.iloc[va] - X.soh_lag1.iloc[va]) ** 2).mean())})
        out = pd.DataFrame(rows)
        out.loc["mean"] = ["", "", out.SVR_RMSE.mean(), out.naive_RMSE.mean()]
        return out

    # ---- Step 2 -------------------------------------------------------------------------
    def recursive_cv_rmse(self, params, train, X, y, groups) -> float:
        """Recursive RMSE: in each fold, forecast the held-out batteries from the 60% origin to the end."""
        fold_rmse = []
        for tr_i, va_i in self.cv.split(X, y, groups):
            fitted = clone(self.make_pipeline()).set_params(**params).fit(X.iloc[tr_i], y.iloc[tr_i])
            err = []
            for bid in groups.iloc[va_i].unique():
                b = train[train.battery_id == bid]
                origin = int(len(b) * ORIGIN_FRACTION)
                err.extend(self.forecaster.forecast(fitted, b, origin) - b.soh.values[origin:])
            fold_rmse.append(np.sqrt(np.mean(np.square(err))))
        return float(np.mean(fold_rmse))

    def naive_recursive_cv_rmse(self, train, X, y, groups) -> float:
        """Naive for E2: keep the last SOH before the origin. Used as the reference."""
        fold_rmse = []
        for _, va_i in self.cv.split(X, y, groups):
            err = []
            for bid in groups.iloc[va_i].unique():
                s = train.soh[train.battery_id == bid].values
                origin = int(len(s) * ORIGIN_FRACTION)
                err.extend(s[origin - 1] - s[origin:])
            fold_rmse.append(np.sqrt(np.mean(np.square(err))))
        return float(np.mean(fold_rmse))

    def select_one_se(self, train, X, y, groups) -> dict:
        """One-SE rule + recursive RMSE. Returns the chosen parameters."""
        res = self.cv_results_
        self.se_best_ = res.iloc[0].cv_RMSE_sd / np.sqrt(self.cfg.n_splits)   # standard error = SD / sqrt(k)
        cand = res[res.cv_RMSE <= res.iloc[0].cv_RMSE + self.se_best_].copy()
        cand["recursive_cv_RMSE"] = [self.recursive_cv_rmse(p, train, X, y, groups) for p in cand.params]
        self.candidates_ = cand.sort_values("recursive_cv_RMSE")
        self.naive_recursive_ = self.naive_recursive_cv_rmse(train, X, y, groups)
        self.naive_one_step_ = float(np.mean([np.sqrt(((y.iloc[va] - X.soh_lag1.iloc[va]) ** 2).mean())
                                              for _, va in self.cv.split(X, y, groups)]))
        self.chosen_ = self.candidates_.iloc[0].params
        return self.chosen_

    def edge_check(self, params: dict) -> pd.DataFrame:
        """Is each parameter strictly inside the grid? On the edge means the grid may need extending."""
        rows = []
        for p in ["svr__C", "svr__gamma"]:
            numeric = [v for v in self.cfg.param_grid[p] if v != "scale"]
            inside = params[p] != "scale" and min(numeric) < params[p] < max(numeric)
            rows.append({"param": p, "value": params[p], "inside_grid": inside})
        return pd.DataFrame(rows)

    def ridge_check(self, train, X, y, groups, extra) -> pd.DataFrame:
        """Edge check beyond the grid (training data only): one-step and recursive CV RMSE of extra settings.

        Used when the chosen gamma sits on the edge of the grid, to see whether a smaller gamma
        (with a matching larger C) would change the choice.
        """
        rows = []
        for p in extra:
            scores = []
            for tr_i, va_i in self.cv.split(X, y, groups):
                m = clone(self.make_pipeline()).set_params(**p).fit(X.iloc[tr_i], y.iloc[tr_i])
                scores.append(np.sqrt(np.mean((m.predict(X.iloc[va_i]) - y.iloc[va_i]) ** 2)))
            rows.append({**p, "cv_RMSE": float(np.mean(scores)),
                         "recursive_cv_RMSE": self.recursive_cv_rmse(p, train, X, y, groups)})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 5. Final model and test-set forecasts
# ----------------------------------------------------------------------------------------
class SVRForecaster:
    """Fits the final model on all training rows, then forecasts the 213 is_eval rows of the 4 test batteries.

    E1: one step ahead, lags are the TRUE SOH of previous cycles. Has a 95% (conformal) interval.
    E2: stands at the first is_eval row (60% origin) and forecasts recursively to the end.
    """

    def __init__(self, config: SVRConfig, builder: FeatureBuilder, forecaster: RecursiveForecaster):
        self.cfg = config
        self.builder = builder
        self.forecaster = forecaster

    def fit(self, X, y, groups, params: dict):
        self.model_ = clone(SVRTuner.make_pipeline()).set_params(**params).fit(X, y)
        # Conformal interval: out-of-fold residuals on the training batteries (same GroupKFold),
        # q = 95th percentile of |error|. Interval = y_hat ± q.
        oof = cross_val_predict(self.model_, X, y, groups=groups, cv=GroupKFold(self.cfg.n_splits))
        self.q_ = float(np.quantile(np.abs(y - oof), self.cfg.interval_level))
        return self

    def predict_e1(self, test: pd.DataFrame) -> pd.DataFrame:
        X_te = self.builder.build(test)
        ev = test.is_eval.values
        assert X_te.loc[ev, self.builder.lag_cols].notna().all().all(), "rows to forecast must have all 5 lags"
        assert X_te.loc[ev, self.builder.feature_names].notna().all().all(), "rows to forecast must have no missing feature"
        y_hat = self.model_.predict(X_te[ev][self.builder.feature_names])
        rows = test.loc[ev, KEYS + ["soh"]].reset_index(drop=True)
        return rows.assign(model="svr", scenario="E1", y_true=rows.soh, y_pred=y_hat,
                           y_lower=y_hat - self.q_, y_upper=y_hat + self.q_)[PRED_COLS]

    def predict_e2(self, test: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for bid, b in test.groupby("battery_id", sort=True):
            b = b.reset_index(drop=True)
            origin = int(np.flatnonzero(b.is_eval)[0])
            y_hat = self.forecaster.forecast(self.model_, b, origin)
            parts.append(b.loc[origin:, KEYS + ["soh"]].assign(y_pred=y_hat))
        rows = pd.concat(parts, ignore_index=True)
        # No interval for E2 (the error grows with the horizon); left empty as the shared format allows
        return rows.assign(model="svr", scenario="E2", y_true=rows.soh,
                           y_lower=np.nan, y_upper=np.nan)[PRED_COLS]

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """E1 and E2 together, in the shared format reports/predictions/<model>.csv."""
        return pd.concat([self.predict_e1(test), self.predict_e2(test)], ignore_index=True)

    @staticmethod
    def interval_coverage(pred: pd.DataFrame) -> float:
        """Share of true values inside the E1 interval (ideally about 95%)."""
        e1 = pred[pred.scenario == "E1"]
        return float(((e1.y_true >= e1.y_lower) & (e1.y_true <= e1.y_upper)).mean())


# ----------------------------------------------------------------------------------------
# 6. Which features matter (RQ4)
# ----------------------------------------------------------------------------------------
class ImportanceAnalyzer:
    """Permutation importance on the E1 rows of the test set.

    Shuffle one feature (breaking its link with SOH) and measure how much the RMSE grows.
    A large increase = the model relies a lot on that feature.
    The 5 lags and soh_roll5 carry almost the same information: shuffling one of them alone lets
    the model read it from the others. So we also shuffle the WHOLE SOH-history group at once.
    """

    def __init__(self, config: SVRConfig, builder: FeatureBuilder):
        self.cfg = config
        self.builder = builder

    def run(self, model, test: pd.DataFrame) -> pd.DataFrame:
        ev = test.is_eval.values
        X_ev = self.builder.build(test)[ev][self.builder.feature_names]
        y_ev = test.soh[ev]

        # Each feature on its own (built-in sklearn function)
        pi = permutation_importance(model, X_ev, y_ev, scoring="neg_root_mean_squared_error",
                                    n_repeats=self.cfg.n_repeats, random_state=self.cfg.seed)
        imp = pd.DataFrame({"feature": self.builder.feature_names,
                            "RMSE_increase": pi.importances_mean, "sd": pi.importances_std})

        # The whole SOH-history group: the SAME row permutation for every column of the group
        rng = np.random.default_rng(self.cfg.seed)
        self.base_rmse_ = float(np.sqrt(((model.predict(X_ev) - y_ev) ** 2).mean()))
        hist_cols = self.builder.lag_cols + ["soh_roll5"]
        inc = []
        for _ in range(self.cfg.n_repeats):
            Xp = X_ev.copy()
            Xp[hist_cols] = X_ev[hist_cols].values[rng.permutation(len(X_ev))]
            inc.append(np.sqrt(((model.predict(Xp) - y_ev) ** 2).mean()) - self.base_rmse_)
        group = pd.DataFrame([{"feature": "[group] SOH history (lags 1-5 + roll5)",
                               "RMSE_increase": np.mean(inc), "sd": np.std(inc)}])
        return pd.concat([imp, group]).sort_values("RMSE_increase", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------------------------
class SVRPlotter:
    """Three figures for the report, saved in reports/figures/."""

    def __init__(self, config: SVRConfig):
        self.cfg = config

    def forecasts(self, test: pd.DataFrame, pred: pd.DataFrame):
        """4 panels, one per test battery: true SOH, E1 forecast (with 95% interval), E2 forecast, EOL line."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        for ax, (bid, b) in zip(axes.ravel(), test.groupby("battery_id", sort=True)):
            p1 = pred[(pred.battery_id == bid) & (pred.scenario == "E1")]
            p2 = pred[(pred.battery_id == bid) & (pred.scenario == "E2")]
            ax.plot(b.cycle_valid, b.soh, color="0.3", lw=1.2, label="true SOH")
            ax.fill_between(p1.cycle_valid, p1.y_lower, p1.y_upper, color="tab:blue", alpha=0.15,
                            label="E1 95% interval")
            ax.plot(p1.cycle_valid, p1.y_pred, color="tab:blue", lw=1.2, label="E1 one step ahead")
            ax.plot(p2.cycle_valid, p2.y_pred, color="tab:orange", lw=1.8, ls="--", label="E2 from the origin")
            ax.axvline(p1.cycle_valid.min() - 0.5, color="0.6", ls=":", lw=1)    # 60% origin
            ax.axhline(EOL_SOH, color="tab:red", lw=0.8, ls="--")                # SOH = 0.8
            c = b.iloc[0]
            ax.set_title(f"{bid}: {c.ambient_C:.0f}°C, {c.I_level_A:.0f} A, cutoff {c.discharge_cutoff_V} V")
            ax.set_xlabel("cycle_valid")
            ax.set_ylabel("SOH")
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle("SVR on the test batteries (dotted line = origin at 60%, red line = EOL at SOH 0.8)")
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "svr_forecasts.png", dpi=150)
        return fig

    def tuning_heatmap(self, tuner):
        """Heatmap of the tuning results: C on the x axis, gamma on the y axis.

        Each cell shows the best epsilon for that (C, gamma) pair.
        Left: one-step-ahead RMSE (round 2). Black border = settings within one SE.
        Right: recursive RMSE, computed only for the settings within one SE (round 3).
        Gold border = chosen setting. Dot = value of the original Go.md grid.
        """
        Cs = self.cfg.param_grid["svr__C"]
        # sklearn's 'scale' = 1 / (n_features × variance), which is ≈ 1/14 ≈ 0.07 after scaling,
        # so it is placed between 0.01 and 0.1 to keep the gamma axis increasing from top to bottom
        n_feat = len(self.cfg.param_grid) and len(tuner.builder.feature_names)
        order = sorted(self.cfg.param_grid["svr__gamma"], key=lambda g: 1 / n_feat if g == "scale" else g)
        gammas = order
        g_labels = [str(g) for g in gammas]
        g_ticks = [f"scale (≈{1 / n_feat:.2f})" if g == "scale" else str(g) for g in gammas]
        res = tuner.cv_results_.assign(g=tuner.cv_results_.param_svr__gamma.astype(str))
        one_step = res.groupby(["g", "param_svr__C"]).cv_RMSE.min().unstack().reindex(index=g_labels, columns=Cs)
        cand = tuner.candidates_.assign(g=tuner.candidates_.param_svr__gamma.astype(str))
        recursive = cand.groupby(["g", "param_svr__C"]).recursive_cv_RMSE.min().unstack().reindex(index=g_labels, columns=Cs)
        in_1se = recursive.notna()
        chosen = (g_labels.index(str(tuner.chosen_["svr__gamma"])), Cs.index(tuner.chosen_["svr__C"]))
        og = self.cfg.original_grid

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        panels = [(one_step, "Round 2: one-step-ahead RMSE (GroupKFold)", 0.03, "Blues_r"),
                  (recursive, "Round 3: recursive RMSE (settings within one SE only)", None, "Oranges_r")]
        for ax, (tab, title, vmax, cmap) in zip(axes, panels):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#EEEEEE")                                   # empty cell
            im = ax.imshow(tab.values, cmap=cm, vmax=vmax, aspect="auto")
            cb = fig.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("RMSE (lower = better)" + (f", colour capped at {vmax}" if vmax else ""))
            for i in range(len(g_labels)):
                for j in range(len(Cs)):
                    v = tab.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v * 1000:.1f}", ha="center", va="center", fontsize=7.5,
                                color="white" if (vmax and v < 0.016) or (not vmax and v < 0.06) else "0.15")
                    if in_1se.values[i, j]:                         # within one SE: black border
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6))
                    if Cs[j] in og["svr__C"] and gammas[i] in og["svr__gamma"]:
                        ax.plot(j, i - 0.30, "o", ms=3, color="0.2")            # original Go.md grid (dot above the number)
            # chosen cell: thick gold border
            ax.add_patch(plt.Rectangle((chosen[1] - 0.5, chosen[0] - 0.5), 1, 1, fill=False, ec="gold", lw=4, zorder=5))
            ax.set_xticks(range(len(Cs)), [str(c) for c in Cs])
            ax.set_yticks(range(len(g_labels)), g_ticks)
            ax.set_xlabel("C (penalty)")
            ax.set_ylabel("gamma (RBF kernel locality)")
            ax.set_title(title, fontsize=11)
        c = tuner.chosen_
        fig.suptitle(f"SVR tuning. Gold border = chosen: C = {c['svr__C']}, epsilon = {c['svr__epsilon']}, "
                     f"gamma = {c['svr__gamma']}\n"
                     "Cell value = RMSE × 1000 (best epsilon of the cell) · black border = within one SE · "
                     f"dot = original Go.md grid · naive: one-step {tuner.naive_one_step_ * 1000:.1f}, "
                     f"recursive {tuner.naive_recursive_ * 1000:.1f}", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "svr_tuning_heatmap.png", dpi=150)
        return fig

    def importance(self, imp: pd.DataFrame):
        """Horizontal bar chart; the orange bar is the SOH-history group."""
        fig, ax = plt.subplots(figsize=(8, 5))
        d = imp.iloc[::-1]
        colors = ["tab:orange" if f.startswith("[group]") else "tab:blue" for f in d.feature]
        ax.barh(d.feature, d.RMSE_increase, xerr=d.sd, color=colors)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel("increase in test RMSE when the feature is shuffled (E1)")
        ax.set_title(f"SVR permutation importance ({self.cfg.n_repeats} repeats)")
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "svr_permutation_importance.png", dpi=150)
        return fig


# ----------------------------------------------------------------------------------------
# 8. Full run
# ----------------------------------------------------------------------------------------
class SVRExperiment:
    """Calls the classes above in order. Each step is its own method so the notebook can run
    them one by one and show intermediate results; run() does everything at once."""

    def __init__(self, config: SVRConfig = None):
        self.cfg = config or SVRConfig()
        self.builder = FeatureBuilder(self.cfg)
        self.recursive = RecursiveForecaster(self.builder)
        self.tuner = SVRTuner(self.cfg, self.builder, self.recursive)
        self.forecaster = SVRForecaster(self.cfg, self.builder, self.recursive)
        self.importance = ImportanceAnalyzer(self.cfg, self.builder)
        self.plotter = SVRPlotter(self.cfg)

    def load_data(self):
        """Step 2: the team's shared data and split."""
        self.train, self.test = load_split()
        self.builder.check_no_leakage()
        self.X_tr, self.y_tr, self.g_tr = self.builder.training_matrix(self.train)
        return self.train, self.test

    def tune(self) -> dict:
        """Step 4: hyperparameter selection (training data only)."""
        self.tuner.grid_search(self.X_tr, self.y_tr, self.g_tr)
        self.params = self.tuner.select_one_se(self.train, self.X_tr, self.y_tr, self.g_tr)
        return self.params

    def fit_predict(self) -> pd.DataFrame:
        """Step 5: fit the final model, forecast E1/E2."""
        self.forecaster.fit(self.X_tr, self.y_tr, self.g_tr, self.params)
        self.pred = self.forecaster.predict(self.test)
        return self.pred

    def evaluate(self):
        """Step 6: score with the team's evaluate(), together with the naive baseline."""
        self.metrics, self.rul = evaluate(pd.concat([self.pred, naive(self.test)], ignore_index=True))
        return self.metrics, self.rul

    def pooled_metrics(self) -> pd.DataFrame:
        m = self.metrics[self.metrics.battery_id == "all"].drop(columns="battery_id")
        return m.set_index(["scenario", "model"]).sort_index()

    def per_battery_rmse(self) -> pd.DataFrame:
        m = self.metrics[self.metrics.battery_id != "all"]
        t = m.pivot_table(index=["scenario", "battery_id"], columns="model", values="RMSE")
        t["svr_better"] = t.svr < t.naive
        return t

    def explain(self) -> pd.DataFrame:
        """RQ4: permutation importance."""
        self.imp = self.importance.run(self.forecaster.model_, self.test)
        return self.imp

    def save(self) -> Path:
        """Saves the prediction file and checks the saved file with the team's evaluate()."""
        self.pred.to_csv(self.cfg.pred_path, index=False)
        evaluate(pd.read_csv(self.cfg.pred_path))
        return self.cfg.pred_path

    def run(self):
        self.load_data()
        self.tune()
        self.fit_predict()
        self.evaluate()
        self.explain()
        self.plotter.tuning_heatmap(self.tuner)
        self.plotter.forecasts(self.test, self.pred)
        self.plotter.importance(self.imp)
        self.save()
        return self


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")      # Windows consoles: allow °C, → and other symbols
    pd.set_option("display.width", 140)
    exp = SVRExperiment().run()
    print("training rows:", len(exp.y_tr))
    print("chosen parameters:", exp.params)
    print(exp.pooled_metrics().round(4))
    print(exp.rul.to_string(index=False))
    print(exp.imp.head(6).round(5).to_string(index=False))
    print("saved", exp.cfg.pred_path.relative_to(ROOT).as_posix())
