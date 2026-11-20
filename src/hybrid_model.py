"""Fifth model: hierarchical Bayesian degradation curve anchored to the latest observation.

Design pre-registered in docs/Rec.md section 8 before any test cell was scored.

    SOH(t) = c - a * t**b,  theta = (c, ln a, ln b),  t = cycle_valid
    prior     : Gaussian over theta, mean/cov of least-squares fits on the training cells' full series
    per cell  : MAP of  W * SSE / s2 + (theta - mu)' Sigma^-1 (theta - mu)  on the cell's history before the origin
    E2        : MAP curve shifted so it passes through the mean of the last ANCHOR_N observed SOH at the origin
    E1        : previous observed SOH + (curve(t) - curve(t-1))
    intervals : E1 +- 95th percentile of |error| from training-cell CV; E2 from Laplace posterior samples + noise
    P(EOL)    : share of anchored posterior curves below 0.8 within the next H cycles

The model mixes three sources: the population curve of the training cells, the cell's own history, and
persistence (the anchor). Run from the repository root: python src/hybrid_model.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import protocol as P

W = 0.03            # weight of the cell's own data (Rec.md 8 #3), chosen by training-cell CV
ANCHOR_N = 5        # observed SOH values averaged for the anchor (Rec.md 8 #4)
H = 20              # alarm horizon in cycles (Rec.md 6)
N_SAMPLES = 2000
SEED = 42
P_STAR = 0.5        # probability threshold of the replacement policy (Rec.md 8 #9)
COST_RATIOS = (10, 20, 50, 100)


def curve(theta, t):
    c, la, lb = theta
    return c - np.exp(la) * np.asarray(t, float) ** np.exp(lb)


def fit_free(t, y):
    """Unpenalised least squares; several starts because the power-law surface has flat valleys."""
    sse = lambda th: np.sum((y - curve(th, t)) ** 2)
    fits = [minimize(sse, (1.0, np.log(1e-3), lb0), method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12)) for lb0 in (-0.7, 0.0, 0.4)]
    return min(fits, key=lambda r: r.fun).x


def make_prior(cells):
    """Mean and covariance of theta over the given cells' full series (a small ridge keeps it invertible)."""
    thetas = np.array([fit_free(g.cycle_valid.to_numpy(float), g.soh.to_numpy()) for _, g in cells.groupby("battery_id")])
    return thetas.mean(0), np.cov(thetas.T) + np.eye(3) * 1e-6


def hessian(f, x, h=1e-4):
    n = len(x)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            out[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return (out + out.T) / 2


def fit_cell(t, y, mu, cov):
    """MAP theta, Laplace covariance, noise variance and anchor level of one cell's history."""
    prec = np.linalg.inv(cov)
    s2 = np.var(y - curve(fit_free(t, y), t))
    obj = lambda th: W * np.sum((y - curve(th, t)) ** 2) / s2 + (th - mu) @ prec @ (th - mu)
    theta = minimize(obj, mu, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-9, fatol=1e-12)).x
    post_cov = np.linalg.inv(hessian(obj, theta) / 2)        # obj = 2 x negative log posterior
    post_cov = (post_cov + post_cov.T) / 2
    if np.linalg.eigvalsh(post_cov).min() <= 0:              # numerical safety: fall back to the prior covariance
        post_cov = cov
    return dict(theta=theta, post_cov=post_cov, s2=s2, t_origin=t[-1], level=np.mean(y[-ANCHOR_N:]))


def anchored(theta, fit, t):
    return curve(theta, t) - curve(theta, [fit["t_origin"]]) + fit["level"]


def forecast_cell(g, mu, cov, rng, q_e1=None):
    """g: one cell sorted by cycle, with is_eval. Returns E1/E2 rows in PRED_COLS and per-cycle P(EOL)."""
    hist, ev = g[~g.is_eval], g[g.is_eval]
    t, y, te = hist.cycle_valid.to_numpy(float), hist.soh.to_numpy(), ev.cycle_valid.to_numpy(float)
    fit = fit_cell(t, y, mu, cov)
    thetas = rng.multivariate_normal(fit["theta"], fit["post_cov"], N_SAMPLES)
    paths = np.array([anchored(th, fit, te) for th in thetas])           # (N_SAMPLES, n_eval)
    noisy = paths + rng.normal(0, np.sqrt(fit["s2"]), paths.shape)
    e2 = anchored(fit["theta"], fit, te)
    # E1: previous observed SOH (history included) plus the curve step between the two cycles
    all_t, all_y = g.cycle_valid.to_numpy(float), g.soh.to_numpy()
    k = len(hist)
    step = curve(fit["theta"], all_t[k:]) - curve(fit["theta"], all_t[k - 1:-1])
    e1 = all_y[k - 1:-1] + step
    q = np.nan if q_e1 is None else q_e1
    base = dict(battery_id=g.battery_id.iloc[0], cycle_valid=ev.cycle_valid.to_numpy(), y_true=ev.soh.to_numpy())
    rows = pd.concat([
        pd.DataFrame(dict(model="hybrid", scenario="E1", y_pred=e1, y_lower=e1 - q, y_upper=e1 + q, **base)),
        pd.DataFrame(dict(model="hybrid", scenario="E2", y_pred=e2, y_lower=np.quantile(noisy, 0.025, 0),
                          y_upper=np.quantile(noisy, 0.975, 0), **base)),
    ], ignore_index=True)[P.PRED_COLS]
    # first predicted crossing of each posterior path (inf if it never crosses within the observed horizon)
    below = paths < P.EOL_SOH
    first = np.where(below.any(1), te[np.argmax(below, 1)], np.inf)
    p_eol = pd.DataFrame(dict(battery_id=g.battery_id.iloc[0], cycle_valid=ev.cycle_valid.to_numpy(),
                              p_eol=[(first <= c + H).mean() for c in te]))
    return rows, p_eol, fit


def with_origin(df):
    df = df.copy()
    pos = df.groupby("battery_id").cumcount()
    n = df.groupby("battery_id").battery_id.transform("size")
    df["is_eval"] = pos >= (n * P.ORIGIN_FRACTION).astype(int)
    return df


def cv_train(train):
    """Leave-one-cell-out on the training cells with a pseudo 60% origin (the design evidence of Rec.md 8)."""
    train = with_origin(train)
    rng = np.random.default_rng(SEED)
    parts, probs = [], []
    for bid, g in train.groupby("battery_id"):
        mu, cov = make_prior(train[train.battery_id != bid])
        rows, p_eol, _ = forecast_cell(g, mu, cov, rng)
        parts.append(rows); probs.append(p_eol)
    return train, pd.concat(parts, ignore_index=True), pd.concat(probs, ignore_index=True)


def e1_quantile(cv_pred):
    e1 = cv_pred[cv_pred.scenario == "E1"]
    return float(np.quantile((e1.y_pred - e1.y_true).abs(), 0.95))


def predict_test(train, test, q_e1):
    mu, cov = make_prior(train)
    rng = np.random.default_rng(SEED)
    parts, probs, fits = [], [], {}
    for bid, g in test.groupby("battery_id"):
        rows, p_eol, fit = forecast_cell(g, mu, cov, rng, q_e1)
        parts.append(rows); probs.append(p_eol); fits[bid] = fit
    return pd.concat(parts, ignore_index=True), pd.concat(probs, ignore_index=True), fits


def diebold_mariano(df):
    """DM test on the loss difference column d (> 0: the hybrid has the smaller |error|).
    Newey-West variance; autocovariances are only taken within a cell."""
    d_all = []
    for _, g in df.groupby("battery_id", sort=True):
        d_all.append(g.d.to_numpy())
    n = sum(len(d) for d in d_all)
    dbar = np.concatenate(d_all).mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lrv = 0.0
    for lag in range(L + 1):
        gam = sum(np.sum((d[lag:] - dbar) * (d[:len(d) - lag] - dbar)) for d in d_all if len(d) > lag) / n
        lrv += gam if lag == 0 else 2 * (1 - lag / (L + 1)) * gam
    stat = dbar / np.sqrt(lrv / n)
    return dict(mean_diff=dbar, DM=stat, p=2 * (1 - norm.cdf(abs(stat))), n=n, lag=L)


def dm_table(pred, against=("naive", "lstm")):
    out = []
    for scenario in ("E1", "E2"):
        h = pred[(pred.model == "hybrid") & (pred.scenario == scenario)].set_index(P.KEYS)
        for other in against:
            o = pred[(pred.model == other) & (pred.scenario == scenario)].set_index(P.KEYS)
            j = h.join(o, rsuffix="_o")
            j["d"] = (j.y_pred_o - j.y_true).abs() - (j.y_pred - j.y_true).abs()
            out.append(dict(scenario=scenario, hybrid_vs=other, **diebold_mariano(j.reset_index())))
    return pd.DataFrame(out)


def policy_cost(alarm, eol_true, ratio):
    """alarm: boolean Series indexed by cycle_valid (eval cycles). Replace at the first alarm before EOL."""
    early = alarm[(alarm.index < eol_true) & alarm.to_numpy()]
    if len(early):
        t_r = early.index[0]
        return dict(replace_at=t_r, wasted_cycles=eol_true - t_r, failure=False, cost=eol_true - t_r)
    return dict(replace_at=np.nan, wasted_cycles=0, failure=True, cost=ratio)


def cost_table(pred, p_eol, test, ratios=COST_RATIOS, p_star=P_STAR):
    """Rec.md 8 #9. Point-forecast alarms for every model; hybrid also with the probability threshold."""
    rows = []
    for bid, g in test.groupby("battery_id"):
        known, ev = g[~g.is_eval], g[g.is_eval]
        if not np.isnan(P.first_below(known.cycle_valid, known.soh)):
            continue                                            # B0047 crossed before the origin
        eol_true = P.first_below(ev.cycle_valid, ev.soh)
        cycles = ev.cycle_valid.to_numpy()
        alarms = {}
        for m, p in pred[(pred.scenario == "E2") & (pred.battery_id == bid)].groupby("model"):
            p = p.sort_values("cycle_valid")
            eol_pred = P.first_below(p.cycle_valid, p.y_pred)
            alarms[m] = pd.Series(eol_pred <= cycles + H, index=cycles)
        pe = p_eol[p_eol.battery_id == bid].set_index("cycle_valid").p_eol.reindex(cycles)
        alarms["hybrid_p*"] = pe >= p_star
        for m, a in alarms.items():
            for r in ratios:
                rows.append(dict(model=m, battery_id=bid, eol_true=eol_true, ratio=r, **policy_cost(a, eol_true, r)))
    return pd.DataFrame(rows)


def self_check(train, test):
    """Fast checks: no test row in the prior, schema, intervals, probabilities, cost rules."""
    assert set(train.battery_id) == set(P.TRAIN_IDS) and set(test.battery_id) == set(P.TEST_IDS)
    a = pd.Series([False, True, True], index=[10, 11, 12])
    assert policy_cost(a, 15, 50) == dict(replace_at=11, wasted_cycles=4, failure=False, cost=4)
    assert policy_cost(a & False, 15, 50)["cost"] == 50
    g = with_origin(train)
    g = g[g.battery_id == "B0005"]
    mu, cov = make_prior(train[train.battery_id != "B0005"])
    rows, p_eol, fit = forecast_cell(g, mu, cov, np.random.default_rng(0), 0.01)
    e2 = rows[rows.scenario == "E2"]
    assert (e2.y_lower <= e2.y_pred).all() and (e2.y_pred <= e2.y_upper).all()
    assert p_eol.p_eol.between(0, 1).all() and p_eol.p_eol.is_monotonic_increasing   # later cycle, wider window
    assert abs(anchored(fit["theta"], fit, [fit["t_origin"]])[0] - fit["level"]) < 1e-12


def cv_summary(cv_train_df, cv_pred):
    """Hybrid vs naive on the training cells' pseudo-origin (design evidence, Rec.md 8)."""
    import compare_models as C
    ev = cv_train_df[cv_train_df.is_eval]
    prev = cv_train_df.groupby("battery_id").soh.shift()[cv_train_df.is_eval]
    held = cv_train_df.soh.where(~cv_train_df.is_eval).groupby(cv_train_df.battery_id).transform("last")[cv_train_df.is_eval]
    naive = pd.concat([ev[P.KEYS].assign(model="naive", scenario=s, y_true=ev.soh, y_pred=y, y_lower=np.nan,
                                         y_upper=np.nan)[P.PRED_COLS] for s, y in (("E1", prev), ("E2", held))])
    both = pd.concat([cv_pred, naive], ignore_index=True)
    rows = []
    for (m, s), g in both.groupby(["model", "scenario"]):
        rows.append(dict(model=m, scenario=s, n=len(g), **P.scores(g).drop("n").to_dict()))
    old = C.MODELS
    C.MODELS = ["naive", "hybrid"]
    alarm = C.confusion(C.alarm_rows(both, cv_train_df, H))
    C.MODELS = old
    return pd.DataFrame(rows), alarm, both


def loco_all12(train, test):
    """Robustness check added after the test run: the frozen design (no re-tuning), leave-one-cell-out over all
    12 tier-A cells, prior from the other 11. Supporting evidence only: W was chosen on 8 of these cells."""
    cells = pd.concat([train, test.drop(columns="is_eval")], ignore_index=True).sort_values(P.KEYS)
    cv_df, cv_pred, _ = cv_train(cells)
    metrics, alarm, both = cv_summary(cv_df, cv_pred)
    per_cell = both.assign(ae=(both.y_pred - both.y_true).abs()).pivot_table(
        index=["scenario", "battery_id"], columns="model", values="ae").reset_index()
    per_cell["hybrid_better"] = per_cell.hybrid < per_cell.naive
    dm = dm_table(both, against=("naive",))
    return metrics, alarm, per_cell, dm


def plot_test(pred, p_eol, test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), gridspec_kw=dict(height_ratios=[2.2, 1]), sharex="col")
    for j, (bid, g) in enumerate(test.groupby("battery_id")):
        ax, bx = axes[0, j], axes[1, j]
        e2 = pred[(pred.scenario == "E2") & (pred.battery_id == bid)]
        ax.fill_between(e2.cycle_valid, e2.y_lower, e2.y_upper, color="#008300", alpha=0.15, lw=0, label="E2 95% interval")
        ax.plot(g.cycle_valid, g.soh, color="#0b0b0b", lw=1.4, label="Observed SOH")
        ax.plot(e2.cycle_valid, e2.y_pred, color="#008300", lw=1.8, label="Hybrid E2")
        ax.axhline(P.EOL_SOH, color="#52514e", lw=0.8, ls=(0, (4, 3)))
        ax.axvline(e2.cycle_valid.min(), color="#52514e", lw=0.8, ls=":")
        ax.set_title(bid, fontsize=10)
        ax.set_ylim(g.soh.min() - 0.1, g.soh.max() + 0.02)     # the lower E2 band runs off-axis (Rec.md 8 #6)
        pe = p_eol[p_eol.battery_id == bid]
        bx.plot(pe.cycle_valid, pe.p_eol, color="#008300", lw=1.6)
        bx.axhline(P_STAR, color="#52514e", lw=0.8, ls=(0, (4, 3)))
        eol = P.first_below(g[g.is_eval].cycle_valid, g[g.is_eval].soh)
        crossed_before = not np.isnan(P.first_below(g[~g.is_eval].cycle_valid, g[~g.is_eval].soh))
        if crossed_before:
            bx.text(0.03, 0.08, "below 0.8 before the origin:\nno RUL (Rec.md 6.1 #3)", transform=bx.transAxes, fontsize=8, color="#52514e")
        elif not np.isnan(eol):
            bx.axvline(eol, color="#d03b3b", lw=1, label="True EOL")
        bx.set(ylim=(-0.03, 1.03), xlabel="Valid cycle")
        for a in (ax, bx):
            a.spines[["top", "right"]].set_visible(False)
            a.grid(axis="y", color="#e6e5e0", lw=0.6)
    axes[0, 0].set_ylabel("SOH")
    axes[1, 0].set_ylabel(f"P(EOL within {H} cycles)")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"Hybrid model on the test cells: E2 forecast from the 60% origin (dotted) and P(EOL); dashed lines: SOH 0.8, p* = {P_STAR}"
                 "\nThe lower edge of the E2 interval is cut by the axis where it falls below the plotted range",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run():
    """Training-cell CV, the single test run, and every output file of the hybrid model."""
    import glob
    train, test = P.load_split()
    self_check(train, test)
    cv_df, cv_pred, _ = cv_train(train)
    q = e1_quantile(cv_pred)
    cv_metrics, cv_alarm, _ = cv_summary(cv_df, cv_pred)
    pred, p_eol, fits = predict_test(train, test, q)
    P.check_pred(pred, test)
    pred.to_csv(P.PRED / "hybrid.csv", index=False)
    others = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(P.PRED / "*.csv")))], ignore_index=True)
    dm = dm_table(others)
    cost = cost_table(others, p_eol, test)
    tables = P.ROOT / "reports" / "tables"
    cv_metrics.to_csv(tables / "hybrid_cv_metrics.csv", index=False)
    cv_alarm.to_csv(tables / "hybrid_cv_alarm.csv", index=False)
    p_eol.to_csv(tables / "hybrid_p_eol.csv", index=False)
    dm.to_csv(tables / "hybrid_dm_tests.csv", index=False)
    cost.to_csv(tables / "hybrid_cost.csv", index=False)
    pd.DataFrame([dict(battery_id=b, c=f["theta"][0], a=np.exp(f["theta"][1]), b=np.exp(f["theta"][2]),
                       noise_sd=np.sqrt(f["s2"]), anchor_level=f["level"]) for b, f in fits.items()]
                 ).to_csv(tables / "hybrid_parameters.csv", index=False)
    plot_test(pred, p_eol, test, P.ROOT / "reports" / "figures" / "hybrid_test.png")
    l_metrics, l_alarm, l_cells, l_dm = loco_all12(train, test)
    l_metrics.to_csv(tables / "hybrid_loco12_metrics.csv", index=False)
    l_alarm.to_csv(tables / "hybrid_loco12_alarm.csv", index=False)
    l_cells.to_csv(tables / "hybrid_loco12_by_cell.csv", index=False)
    l_dm.to_csv(tables / "hybrid_loco12_dm.csv", index=False)
    return dict(q_e1=q, cv_metrics=cv_metrics, cv_alarm=cv_alarm, pred=pred, p_eol=p_eol, dm=dm, cost=cost, fits=fits,
                loco_metrics=l_metrics, loco_alarm=l_alarm, loco_cells=l_cells, loco_dm=l_dm)


if __name__ == "__main__":
    out = run()
    m, rul = P.evaluate(out["pred"])
    print(m[m.battery_id == "all"].round(4).to_string(index=False))
    print(rul.to_string(index=False))
    print("self-checks passed; outputs written")
