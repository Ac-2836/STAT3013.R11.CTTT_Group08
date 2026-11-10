"""Utilities scoped to Quoc Dat's regression and path analysis (not the shared protocol)."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
TABLES = ROOT / 'reports' / 'tables'
FIGURES = ROOT / 'reports' / 'figures'
TRAIN = ('B0005', 'B0006', 'B0018', 'B0033', 'B0045', 'B0046', 'B0048', 'B0056')
TEST = ('B0007', 'B0034', 'B0047', 'B0055')
SCHEMA = ['model', 'battery_id', 'cycle_valid', 'scenario', 'y_true', 'y_pred', 'y_lower', 'y_upper']


def save_table(frame, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / (name + '.csv'), index=False)
    return frame


def load_cycles():
    x = pd.read_csv(DATA / 'cycles_clean.csv')
    assert not x.duplicated(['battery_id', 'cycle_valid']).any()
    assert not x.duplicated(['battery_id', 'op_index']).any()
    assert x[['cap_outlier', 'break_in']].dtypes.eq(bool).all()
    for key in ['chg_op_index', 'eis_op_index']:
        ok = x[key].notna()
        assert (x.loc[ok, key] < x.loc[ok, 'op_index']).all()
    assert (pd.to_datetime(x.chg_end_time) <= pd.to_datetime(x.start_time)).all()
    return x.sort_values(['battery_id', 'cycle_valid']).reset_index(drop=True)


def regression_data(cycles):
    x = cycles.loc[cycles.tier.eq('A') & ~cycles.cap_outlier & ~cycles.break_in].copy()
    required = ['soh', 'cycle_valid', 'ambient_C', 'I_level_A', 'discharge_cutoff_V']
    assert x[required].notna().all().all()
    assert (x.soh > 0).all()
    assert set(x.battery_id) == set(TRAIN) | set(TEST)
    assert set(TRAIN).isdisjoint(TEST)
    x['ln_soh'] = np.log(x.soh)
    return x


def origin_position(n):
    """First forecast row is floor(.6*n); origin is the preceding retained row."""
    k = int(np.floor(.6 * n))
    if not 0 < k < n:
        raise ValueError('Insufficient history/horizon')
    return k


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    assert len(y) and np.isfinite(y).all() and np.isfinite(p).all()
    den = np.sum((y-y.mean())**2)
    return dict(n=len(y), MAE=float(np.mean(abs(y-p))), RMSE=float(np.sqrt(np.mean((y-p)**2))),
                MAPE_pct=float(100*np.mean(abs((y-p)/y))) if (y != 0).all() else np.nan,
                R2=float(1-np.sum((y-p)**2)/den) if den > 0 else np.nan)


def metric_table(pred):
    rows = []
    for (model, scenario), g in pred.groupby(['model', 'scenario'], sort=True):
        per_cell = []
        for bid, cell in g.groupby('battery_id', sort=True):
            m = metrics(cell.y_true, cell.y_pred)
            rows.append(dict(model=model, scenario=scenario, battery_id=bid, aggregation='battery', **m))
            per_cell.append(m)
        rows.append(dict(model=model, scenario=scenario, battery_id='ALL', aggregation='micro',
                         **metrics(g.y_true, g.y_pred)))
        macro = {key: float(np.mean([m[key] for m in per_cell])) for key in ['MAE', 'RMSE', 'MAPE_pct', 'R2']}
        rows.append(dict(model=model, scenario=scenario, battery_id='ALL', aggregation='macro', n=len(g), **macro))
    return pd.DataFrame(rows)


def first_crossing(cycles, values):
    cycles, values = np.asarray(cycles), np.asarray(values)
    hits = cycles[values <= .8]
    return float(hits[0]) if len(hits) else np.nan


def eol_record(history, forecast):
    """First observed crossing is absorbing; never score left/right-censored RUL."""
    observed = pd.concat([history[['cycle_valid', 'soh']],
                          forecast[['cycle_valid', 'y_true']].rename(columns={'y_true': 'soh'})])
    origin = int(history.cycle_valid.iloc[-1])
    true_eol = first_crossing(observed.cycle_valid, observed.soh)
    pred_eol = first_crossing(forecast.cycle_valid, forecast.y_pred)
    if np.isfinite(true_eol) and true_eol <= origin:
        status = 'already_at_or_before_origin'
    elif not np.isfinite(true_eol):
        status = 'observed_right_censored'
    elif not np.isfinite(pred_eol):
        status = 'prediction_no_crossing_in_horizon'
    else:
        status = 'both_cross_within_horizon'
    valid = status == 'both_cross_within_horizon'
    return dict(origin_cycle=origin, horizon_end=int(forecast.cycle_valid.iloc[-1]),
                observed_EOL_cycle=true_eol, predicted_EOL_cycle=pred_eol, status=status,
                observed_RUL=true_eol-origin if np.isfinite(true_eol) and true_eol > origin else np.nan,
                predicted_RUL=pred_eol-origin if np.isfinite(pred_eol) and status != 'already_at_or_before_origin' else np.nan,
                signed_RUL_error=pred_eol-true_eol if valid else np.nan,
                absolute_RUL_error=abs(pred_eol-true_eol) if valid else np.nan)


def validate_predictions(pred, data):
    assert pred.columns.tolist() == SCHEMA
    assert not pred.duplicated(['model', 'battery_id', 'cycle_valid', 'scenario']).any()
    assert set(pred.battery_id) == set(TEST)
    assert set(pred.scenario) == {'E1', 'E2'}
    assert np.isfinite(pred[['y_true', 'y_pred']]).all().all()
    assert pred[['y_lower', 'y_upper']].isna().all().all()
    for (_, bid, _), g in pred.groupby(['model', 'battery_id', 'scenario']):
        x = data[data.battery_id.eq(bid)].sort_values('cycle_valid')
        h = x.iloc[origin_position(len(x)):]
        np.testing.assert_array_equal(g.cycle_valid, h.cycle_valid)
        np.testing.assert_allclose(g.y_true, h.soh, rtol=0, atol=1e-14)


def environment_manifest():
    packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels', 'scikit-learn',
                'semopy', 'nbformat', 'nbclient', 'ipykernel']
    return dict(python=platform.python_version(), seed=42,
                packages={p: importlib.metadata.version(p) for p in packages},
                inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in DATA.glob('*.csv')}, train=list(TRAIN), test=list(TEST))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
