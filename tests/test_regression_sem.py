"""Checks for leakage, censoring and SEM failures that previously went unnoticed."""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
import os
import hashlib
import io
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from regression_sem_utils import (load_cycles, regression_data, TRAIN, TEST, metrics,
                                  eol_record, origin_position, validate_predictions, SCHEMA, ROOT)
from regression_analysis import (fit_forecasters, forecast, center, backward,
                                 operating_condition_check, evaluate_regression)
from sem_analysis import (prepare_sem, fit_sem, effects, PRIMARY, EXOGENOUS,
                          block_bootstrap, run_sem)


# Fast suite (~1-2 min): python -m unittest discover -s tests
# Full suite incl. end-to-end reruns (~10+ min): set RUN_SLOW=1 first.
RUN_SLOW = bool(os.environ.get('RUN_SLOW'))

class EOLTests(unittest.TestCase):
    def record(self, past, actual, prediction):
        h = pd.DataFrame({'cycle_valid':range(1,len(past)+1),'soh':past})
        f = pd.DataFrame({'cycle_valid':range(len(past)+1,len(past)+1+len(actual)),
                          'y_true':actual,'y_pred':prediction})
        return eol_record(h,f)

    def test_exact_threshold_at_origin_invalidates_rul(self):
        r = self.record([.95,.8],[.79,.78],[.82,.79])
        self.assertEqual(r['status'],'already_at_or_before_origin')
        self.assertTrue(np.isnan(r['absolute_RUL_error']))

    def test_historical_crossing_remains_eol_after_recovery(self):
        r = self.record([.9,.79,.88],[.87,.79],[.84,.79])
        self.assertEqual(r['status'],'already_at_or_before_origin')
        self.assertEqual(r['observed_EOL_cycle'],2)

    def test_valid_rul_uses_cycle_coordinates(self):
        h = pd.DataFrame({'cycle_valid':[1,3], 'soh':[1.,.9]})
        f = pd.DataFrame({'cycle_valid':[5,8,9],'y_true':[.85,.8,.79],'y_pred':[.8,.78,.77]})
        r = eol_record(h,f)
        self.assertEqual(r['observed_RUL'],5)
        self.assertEqual(r['predicted_RUL'],2)
        self.assertEqual(r['absolute_RUL_error'],3)

    def test_no_prediction_crossing_retained(self):
        r = self.record([1.,.9],[.81,.8],[.85,.83])
        self.assertEqual(r['status'],'prediction_no_crossing_in_horizon')
        self.assertTrue(np.isnan(r['absolute_RUL_error']))

    def test_observed_right_censoring(self):
        r = self.record([1.,.9],[.85,.81],[.79,.78])
        self.assertEqual(r['status'],'observed_right_censored')
        self.assertTrue(np.isnan(r['absolute_RUL_error']))

    def test_metrics_independent_reference(self):
        y,p = np.array([.91,.85,.79]),np.array([.9,.8,.82])
        m = metrics(y,p)
        self.assertAlmostEqual(m['MAE'],mean_absolute_error(y,p))
        self.assertAlmostEqual(m['RMSE'],np.sqrt(mean_squared_error(y,p)))
        self.assertAlmostEqual(m['R2'],r2_score(y,p))
        self.assertTrue(np.isnan(metrics([.8,.8],[.81,.79])['R2']))


class ModelIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cycles = load_cycles()
        cls.data = regression_data(cls.cycles)
        cls.train = cls.data[cls.data.battery_id.isin(TRAIN)]
        cls.test = cls.data[cls.data.battery_id.isin(TEST)]
        cls.bundle = fit_forecasters(cls.train)
        cls.sem, _ = prepare_sem(cls.cycles, export=False)
        protected = list((ROOT/'data').rglob('*')) + [ROOT/'docs/Rec.md']
        protected += [p for p in (ROOT/'src').glob('*.py') if p.name not in
                      {'regression_analysis.py','sem_analysis.py','regression_sem_utils.py',
                       'run_regression_sem.py'}]
        cls.protected_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in protected if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        for path, digest in cls.protected_hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AssertionError('Protected data/group pipeline changed: '+str(path))

    def test_e2_never_reads_future_outcomes_or_measurements(self):
        original = forecast(self.bundle,self.test)
        changed = self.test.copy()
        for bid,g in changed.groupby('battery_id'):
            future = g.index[origin_position(len(g)):]
            changed.loc[future,['soh','ln_soh','T_max_C','chg_cc_time_s','Re_ohm','Rct_ohm']] = 999
        p = forecast(self.bundle,changed)
        np.testing.assert_allclose(original.loc[original.scenario.eq('E2'),'y_pred'],
                                   p.loc[p.scenario.eq('E2'),'y_pred'],rtol=0,atol=0)

    def test_e1_reads_only_previous_retained_outcome(self):
        g = self.test[self.test.battery_id.eq('B0007')].copy()
        base = forecast(self.bundle,g)
        k = origin_position(len(g))+2
        changed = g.copy(); changed.iloc[k,changed.columns.get_loc('soh')] *= .8
        new = forecast(self.bundle,changed)
        mask = (base.scenario=='E1') & base.model.eq('anchored_loglinear')
        before,after = base.loc[mask,'y_pred'].to_numpy(),new.loc[mask,'y_pred'].to_numpy()
        self.assertEqual(np.flatnonzero(before!=after).tolist(),[3])

    def test_full_tier_a_operating_conditions_are_fixed(self):
        checks = operating_condition_check(self.cycles[self.cycles.tier.eq('A')])
        self.assertEqual(len(checks), 12*3)
        self.assertTrue(checks.fixed.all())
        self.assertTrue(checks.unique_values.eq(1).all())

    def test_unknown_future_operating_conditions_fail_closed(self):
        g = self.test[self.test.battery_id.eq('B0007')].copy()
        for feature in ['ambient_C', 'I_level_A', 'discharge_cutoff_V']:
            for value in [999., np.nan, np.inf]:
                with self.subTest(feature=feature, value=value):
                    changed = g.copy()
                    changed.iloc[origin_position(len(g))+1, changed.columns.get_loc(feature)] = value
                    with self.assertRaisesRegex(ValueError, 'unknown future conditions'):
                        forecast(self.bundle, changed)

    def test_groupkfold_has_no_battery_overlap(self):
        from sklearn.model_selection import GroupKFold
        self.assertEqual(len(TRAIN), 8)
        self.assertEqual(len(TEST), 4)
        self.assertTrue(set(TRAIN).isdisjoint(TEST))
        validation_ids = []
        for it, iv in GroupKFold(4).split(self.train, groups=self.train.battery_id):
            train_ids, val_ids = set(self.train.iloc[it].battery_id), set(self.train.iloc[iv].battery_id)
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertEqual(train_ids | val_ids, set(TRAIN))
            validation_ids.extend(val_ids)
        self.assertCountEqual(validation_ids, TRAIN)

    def test_forecast_is_deterministic(self):
        pd.testing.assert_frame_equal(forecast(self.bundle, self.test), forecast(self.bundle, self.test))

    def isolated_outputs(self, stack, root):
        (root/'reports/predictions').mkdir(parents=True)
        tables = root/'reports/tables'; tables.mkdir()
        figures = root/'reports/figures'; figures.mkdir()
        for module in ['regression_analysis', 'sem_analysis']:
            stack.enter_context(patch(module+'.ROOT', root))
            stack.enter_context(patch(module+'.TABLES', tables))
            stack.enter_context(patch(module+'.FIGURES', figures))
        stack.enter_context(patch('regression_sem_utils.TABLES', tables))
        return tables

    @unittest.skipUnless(RUN_SLOW, 'slow: set RUN_SLOW=1')
    def test_regression_selection_ignores_test_outcomes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); tables = self.isolated_outputs(stack, root)
            stack.enter_context(redirect_stdout(io.StringIO()))
            first = evaluate_regression(self.data)
            prediction = pd.read_csv(root/'reports/predictions/regression.csv')
            validate_predictions(prediction, self.data)
            selected = (tables/'regression_selected.json').read_bytes()
            changed = self.data.copy()
            changed.loc[changed.battery_id.isin(TEST), 'soh'] *= .9
            changed['ln_soh'] = np.log(changed.soh)
            second = evaluate_regression(changed)
            self.assertEqual(first['winners'], second['winners'])
            self.assertEqual(selected, (tables/'regression_selected.json').read_bytes())
            self.assertEqual(first['winners'], {'E1':'anchored_loglinear', 'E2':'anchored_quadratic'})

    @unittest.skipUnless(RUN_SLOW, 'slow: set RUN_SLOW=1')
    def test_full_sem_reproduces_committed_tables(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); tables = self.isolated_outputs(stack, root)
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = run_sem(self.sem)
            self.assertEqual(result['bootstrap']['valid'], 200)
            self.assertEqual(len(result['fit']), 5)
            self.assertEqual(len(result['coefficients']), 12)
            self.assertTrue((root/'reports/figures/sem_path_diagram.png').exists())
            for name, frame in [('sem_fit_comparison', result['fit']),
                                ('sem_effects', result['effects']), ('sem_effect_intervals', result['intervals'])]:
                pd.testing.assert_frame_equal(pd.read_csv(tables/(name+'.csv')),
                    pd.read_csv(ROOT/'reports/tables'/(name+'.csv')), check_exact=False, rtol=1e-12, atol=1e-12)

    def test_bootstrap_preserves_blocks_and_rejects_invalid_fits(self):
        _, good_result, good_paths = fit_sem(self.sem)
        nonfinite = good_paths.copy(); nonfinite.loc[nonfinite.op.eq('~'), 'Estimate'] = np.nan
        boundary = good_paths.copy()
        boundary.loc[(boundary.op=='~~') & (boundary.lval==boundary.rval), 'Estimate'] = 0
        calls = []
        def sampled_fit(sample, **kwargs):
            calls.append(sample.copy())
            if len(calls) == 1:
                raise RuntimeError('Nonconverged test fit')
            if len(calls) == 2:
                return None, good_result, nonfinite
            if len(calls) == 3:
                return None, good_result, boundary
            return None, good_result, good_paths.copy()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); tables = self.isolated_outputs(stack, root)
            stack.enter_context(patch('sem_analysis.fit_sem', side_effect=sampled_fit))
            _, _, summary = block_bootstrap(self.sem, target=2, max_attempts=5)
            self.assertEqual(summary['failures'], 3)
            self.assertEqual(summary['attempts'], 5)
            for sample in calls:
                self.assertEqual(sample.bootstrap_block.nunique(), 22)
                for _, block in sample.groupby('bootstrap_block'):
                    expected = self.sem[self.sem.battery_id.eq(block.battery_id.iloc[0])]
                    pd.testing.assert_frame_equal(block.drop(columns='bootstrap_block').reset_index(drop=True),
                                                  expected.reset_index(drop=True))
            first_ids = pd.read_csv(tables/'sem_bootstrap_audit.csv').sampled_batteries.tolist()
            calls.clear()
            block_bootstrap(self.sem, target=2, max_attempts=5)
            self.assertEqual(first_ids, pd.read_csv(tables/'sem_bootstrap_audit.csv').sampled_batteries.tolist())
            self.assertLess(calls[0].battery_id.nunique(), 22)

    def test_sem_rejects_invalid_covariance_and_optimizer_values(self):
        bad = self.sem.copy(); bad['Re_ohm'] = bad.Rct_ohm
        with self.assertRaisesRegex(ValueError, 'Rank-deficient'):
            fit_sem(bad)
        for objective, values in [(np.nan, [1.]), (0., [np.inf])]:
            with patch('sem_analysis.Model') as constructor:
                constructor.return_value.fit.return_value = SimpleNamespace(success=True, fun=objective, message='test')
                constructor.return_value.param_vals = np.array(values)
                with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                    fit_sem(self.sem)

    def test_bootstrap_exhaustion_raises_and_logs_every_failure(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            tables = self.isolated_outputs(stack, Path(directory))
            mocked = stack.enter_context(patch('sem_analysis.fit_sem', side_effect=RuntimeError('Nonconverged')))
            with self.assertRaisesRegex(RuntimeError, 'Only 0 valid bootstrap replicates after 2 attempts'):
                block_bootstrap(self.sem, target=1, max_attempts=2)
            self.assertEqual(mocked.call_count, 2)
            audit = pd.read_csv(tables/'sem_bootstrap_audit.csv')
            self.assertEqual(len(audit), 2)
            self.assertFalse(audit.valid.any())

    def test_effects_and_bootstrap_paths_match_coefficients(self):
        paths = pd.read_csv(ROOT/'reports/tables/sem_primary_explicit_exog_coefficients.csv')
        pd.testing.assert_frame_equal(effects(paths), pd.read_csv(ROOT/'reports/tables/sem_effects.csv'))
        merged = pd.read_csv(ROOT/'reports/tables/sem_primary_paths_with_bootstrap.csv')
        intervals = pd.read_csv(ROOT/'reports/tables/sem_path_intervals.csv')
        pd.testing.assert_frame_equal(merged[['lval','rval','Estimate','Est. Std']].reset_index(drop=True),
                                      paths[paths.op.eq('~')][['lval','rval','Estimate','Est. Std']].reset_index(drop=True))
        for _, row in merged.iterrows():
            interval = intervals[(intervals.lval==row.lval) & (intervals.rval==row.rval)].iloc[0]
            self.assertEqual(row.bootstrap_CI_low, interval.bootstrap_CI_low)
            self.assertEqual(row.bootstrap_CI_high, interval.bootstrap_CI_high)
        replicates = pd.read_csv(ROOT/'reports/tables/sem_bootstrap_effect_replicates.csv')
        intervals = pd.read_csv(ROOT/'reports/tables/sem_effect_intervals.csv')
        for _, row in intervals.iterrows():
            values = replicates.loc[replicates.predictor.eq(row.predictor), row.effect]
            np.testing.assert_allclose([row.CI_low, row.CI_high], np.quantile(values, [.025,.975]), atol=1e-14)
            self.assertEqual(row.excludes_zero, row.CI_low > 0 or row.CI_high < 0)

    def test_selection_preserves_main_effect_hierarchy(self):
        _,trace = backward(center(self.train)[0])
        for formula in trace.formula:
            terms = formula.split(' ~ ')[1].split(' + ')
            for term in terms:
                if ':' in term:
                    for parent in term.split(':'):
                        self.assertIn(parent,terms)

    def test_sem_uses_actual_eis_not_carried_value(self):
        modified = self.cycles.copy()
        idx = modified.index[modified.battery_id.eq('B0005') & modified.eis_op_index.notna()][0]
        modified.loc[idx,['Re_ohm','Rct_ohm']] = [.4,.4]
        repaired,_ = prepare_sem(modified, export=False)
        keys=['battery_id','eis_op_index']
        pd.testing.assert_frame_equal(repaired[keys+['Re_ohm','Rct_ohm']],self.sem[keys+['Re_ohm','Rct_ohm']])

    def test_sem_nonconvergence_is_rejected(self):
        with patch('sem_analysis.Model') as constructor:
            constructor.return_value.fit.return_value = SimpleNamespace(success=False,message='test failure')
            with self.assertRaisesRegex(RuntimeError,'Nonconverged'):
                fit_sem(self.sem)

    def test_sem_local_identification_and_degrees_of_freedom(self):
        from semopy import calc_stats
        model,_,_ = fit_sem(self.sem)
        self.assertEqual(int(calc_stats(model).loc['Value','DoF']),3)
        self.assertEqual(np.linalg.matrix_rank(model.calc_fim()),len(model.param_vals))

    def test_sem_unstandardized_unit_conversion(self):
        _,_,p = fit_sem(self.sem)
        altered = self.sem.copy(); altered['I_level_A'] *= 1000
        _,_,q = fit_sem(altered)
        a,b = effects(p).set_index('predictor'),effects(q).set_index('predictor')
        np.testing.assert_allclose(a.loc['I_level_A'],1000*b.loc['I_level_A'],rtol=1e-7,atol=1e-9)

    def test_export_schema_and_protocol(self):
        p = pd.read_csv(ROOT/'reports/predictions/regression.csv')
        validate_predictions(p,self.data)

    def test_cv_centers_use_only_fold_training_cells(self):
        import json
        splits = pd.read_csv(ROOT/'reports/tables/regression_cv_splits.csv')
        for _,row in splits.iterrows():
            fit_ids = set(row.training_batteries.split(','))
            validation_ids = set(row.validation_batteries.split(','))
            self.assertTrue(fit_ids.isdisjoint(validation_ids))
            self.assertTrue((fit_ids|validation_ids).isdisjoint(TEST))
            frame = self.train[self.train.battery_id.isin(fit_ids)]
            for col,value in json.loads(row.means).items():
                self.assertAlmostEqual(value,frame[col].mean())

    def test_bootstrap_counts_and_failure_log(self):
        import json
        summary = json.loads((ROOT/'reports/tables/sem_bootstrap_summary.json').read_text())
        audit = pd.read_csv(ROOT/'reports/tables/sem_bootstrap_audit.csv')
        reps = pd.read_csv(ROOT/'reports/tables/sem_bootstrap_effect_replicates.csv')
        self.assertGreaterEqual(summary['valid'],200)
        self.assertEqual(int(audit.valid.sum()),summary['valid'])
        self.assertEqual(int((~audit.valid).sum()),summary['failures'])
        self.assertEqual(len(audit),summary['attempts'])
        self.assertEqual(reps.replicate.nunique(),summary['valid'])
        self.assertTrue(audit.sampled_batteries.str.split(',').map(len).eq(self.sem.battery_id.nunique()).all())

    def test_diagram_values_match_primary_model(self):
        diagram = pd.read_csv(ROOT/'reports/tables/sem_diagram_coefficients.csv')
        paths = pd.read_csv(ROOT/'reports/tables/sem_primary_explicit_exog_coefficients.csv')
        expected = paths[paths.op.eq('~')][diagram.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(diagram,expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
