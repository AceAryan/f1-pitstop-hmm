import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from src.data_loader import load_race
from src.features import FEATURE_COLUMNS, build_observation_matrix, get_stint_records
from src.hmm import GaussianHMM
from src.pipeline import (
    FINAL_TEST_YEAR,
    HMM_STATE_CANDIDATES,
    _get_season_races,
    _feature_columns,
    _first_threshold_entry,
    _match_laps,
    _load_data,
    _pooled_metrics,
    _write_race_explorer_json,
    _write_validation_metrics_json,
    _timing_rows,
    evaluate_predictions,
    run_evaluate,
    run_preprocess,
)


def _lap(
    year: int,
    grand_prix: str,
    driver: str,
    lap: int,
    stint: int,
    lap_time: float,
    *,
    pit_in: bool = False,
    pit_out: bool = False,
    track_status: str = '1',
    accurate: bool = True,
) -> dict:
    return {
        'Year': year,
        'GrandPrix': grand_prix,
        'Driver': driver,
        'LapNumber': lap,
        'Stint': stint,
        'LapTime': lap_time,
        'Sector3Time': 25.0 + lap_time / 100.0,
        'Compound': 'MEDIUM',
        'TyreLife': lap,
        'PitInTime': pd.Timestamp('2022-01-01') if pit_in else pd.NaT,
        'PitOutTime': pd.Timestamp('2022-01-01') if pit_out else pd.NaT,
        'TrackStatus': track_status,
        'IsAccurate': accurate,
    }


class FeaturePipelineTests(unittest.TestCase):
    def test_loader_explains_fastf1_soft_failure_when_laps_are_unavailable(self):
        session = unittest.mock.Mock()
        session.laps.copy.side_effect = DataNotLoadedError(
            'The data you are trying to access has not been loaded yet.'
        )

        with patch('src.data_loader.fastf1.get_session', return_value=session):
            with self.assertRaisesRegex(
                RuntimeError,
                'did not load lap timing data.*FASTF1_DEBUG=1',
            ) as error:
                load_race(2023, 'Singapore Grand Prix')

        self.assertIsInstance(error.exception.__cause__, DataNotLoadedError)
        session.load.assert_called_once_with(
            telemetry=False, weather=False, messages=False
        )

    def test_filters_dirty_laps_and_keeps_race_lap_pit_targets(self):
        rows = []
        for year, base_time in ((2022, 100.0), (2023, 90.0)):
            rows.extend(
                _lap(year, 'Bahrain', 'HAM', lap, 1, base_time + lap)
                for lap in range(1, 4)
            )
            rows.append(_lap(year, 'Bahrain', 'HAM', 4, 1, base_time + 4, pit_in=True))
            rows.append(_lap(year, 'Bahrain', 'HAM', 5, 2, base_time + 5, pit_out=True))
            rows.extend(
                _lap(year, 'Bahrain', 'HAM', lap, 2, base_time + lap)
                for lap in range(6, 10)
            )
        rows.extend([
            _lap(2023, 'Bahrain', 'VER', 1, 1, 200.0),
            _lap(2023, 'Bahrain', 'VER', 2, 1, 201.0, track_status='4'),
            _lap(2023, 'Bahrain', 'VER', 3, 1, 202.0, accurate=False),
            _lap(2023, 'Bahrain', 'VER', 4, 1, 203.0),
            _lap(2023, 'Bahrain', 'VER', 5, 1, 204.0),
        ])

        features = build_observation_matrix(pd.DataFrame(rows))

        self.assertNotIn(4, features.loc[features['Driver'].eq('HAM'), 'LapNumber'].values)
        self.assertNotIn(5, features.loc[features['Driver'].eq('HAM'), 'LapNumber'].values)
        self.assertFalse(features['TrackStatus'].isin(['4']).any())
        self.assertTrue(features['IsAccurate'].all())

        ham = features[features['Driver'].eq('HAM')]
        for year in (2022, 2023):
            first_stint = ham[(ham['Year'] == year) & (ham['Stint'] == 1)]
            self.assertEqual(first_stint['LapNumber'].tolist(), [1, 2, 3])
            self.assertTrue((first_stint['actual_pit_lap'] == 4).all())
            self.assertEqual(first_stint.iloc[0]['lap_time_delta'], 0.0)

        records = get_stint_records(features)
        self.assertEqual(len(records), 5)
        ham_first_laps = {
            (record['Year'], record['Stint']): record['lap_numbers'][0]
            for record in records if record['Driver'] == 'HAM'
        }
        self.assertEqual(ham_first_laps[(2022, 1)], 1)
        self.assertEqual(ham_first_laps[(2023, 1)], 1)

    def test_hmm_uses_ordered_left_to_right_states_and_causal_filter(self):
        observations = np.array([
            [age * 0.2, age * 0.1, age * 0.3, age / 12]
            for age in range(12)
        ], dtype=float)
        model = GaussianHMM(n_states=3, n_iter=3)
        model.fit([observations, observations + np.array([0.1, 0.0, 0.1, 0.0])])

        np.testing.assert_array_equal(model.pi, [1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(model.A.sum(axis=1), 1.0))
        self.assertTrue(np.all(model.A[np.tril_indices(3, -1)] == 0.0))

        short_filter = model.filter(observations[:5])
        full_filter = model.filter(observations)
        np.testing.assert_allclose(short_filter, full_filter[:5])
        self.assertEqual(short_filter.shape, (5, 3))

    def test_hmm_state_count_candidates_cover_two_three_and_four_states(self):
        self.assertEqual(HMM_STATE_CANDIDATES, (2, 3, 4))
        observations = np.array([
            [age * 0.2, age * 0.1, age * 0.3, age / 12]
            for age in range(12)
        ], dtype=float)
        for n_states in HMM_STATE_CANDIDATES:
            model = GaussianHMM(n_states=n_states, n_iter=2)
            model.fit([observations, observations + 0.05])
            probabilities = model.filter(observations)
            self.assertEqual(probabilities.shape, (12, n_states))
            self.assertIsNotNone(model.predict_pit_laps(observations))

    def test_ablation_excludes_tyre_age_from_hmm_features(self):
        self.assertNotIn('tyre_age_norm', _feature_columns(include_tyre_age=False))
        self.assertIn('rolling_degradation', _feature_columns(include_tyre_age=False))

    def test_hmm_alarm_uses_first_posterior_threshold_crossing(self):
        predictions = _first_threshold_entry(
            np.array([0.1, 0.35, 0.7, 0.9]),
            np.array([3, 4, 5, 6]),
            threshold=0.6,
        )

        self.assertEqual(predictions, [5])

    def test_pit_matching_uses_real_lap_numbers(self):
        metrics = evaluate_predictions([72], [72], tolerance=2)
        self.assertEqual(metrics['TP'], 1)
        self.assertEqual(metrics['FP'], 0)
        self.assertEqual(metrics['FN'], 0)

    def test_pit_matching_maximizes_one_to_one_matches(self):
        matches = _match_laps([2, 3], [1, 2], tolerance=1)

        self.assertEqual(len(matches), 2)
        self.assertEqual(set(matches), {(2, 1), (3, 2)})

    def test_pooled_metrics_are_not_a_mean_of_driver_race_scores(self):
        metrics = _pooled_metrics(
            {
                (2023, 'Bahrain', 'VER'): [10, 20],
                (2023, 'Bahrain', 'HAM'): [],
            },
            {
                (2023, 'Bahrain', 'VER'): [10],
                (2023, 'Bahrain', 'HAM'): [30],
            },
            tolerance=2,
        )

        self.assertEqual((metrics['TP'], metrics['FP'], metrics['FN']), (1, 1, 1))
        self.assertEqual(metrics['F1'], 0.5)

    def test_timing_report_pairs_one_to_one_and_keeps_unmatched_predictions(self):
        rows = _timing_rows(
            'HMM',
            (2023, 'Bahrain', 'VER'),
            predicted=[10, 30],
            actual=[11],
        )

        matched = [row for row in rows if row['MatchStatus'] == 'matched']
        unmatched = [row for row in rows if row['MatchStatus'] == 'unmatched_prediction']
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]['PredictedMinusActual'], -1)
        self.assertEqual(len(unmatched), 1)

    def test_static_site_json_exports_validation_curve_and_race_timeline(self):
        config = {
            'hmm_states': 3,
            'hmm_threshold': 0.6,
            'hmm_no_age_states': 2,
            'hmm_no_age_threshold': 0.7,
            'tyre_age_threshold': 0.8,
        }
        rows = []
        for method, n_states, thresholds in (
            ('HMM', 3, [0.3, 0.6]),
            ('HMM', 4, [0.3]),
            ('HMMNoAge', 2, [0.3, 0.7]),
            ('TyreAge', np.nan, [0.6, 0.8]),
            ('Logistic', np.nan, [np.nan]),
        ):
            for threshold in thresholds:
                rows.append({
                    'Method': method,
                    'NStates': n_states,
                    'Threshold': threshold,
                    'Precision': 0.5,
                    'Recall': 0.4,
                    'F1': 0.444,
                    'TP': 2,
                    'FP': 2,
                    'FN': 3,
                })

        with tempfile.TemporaryDirectory() as directory:
            validation_path = Path(directory) / 'validation.json'
            race_path = Path(directory) / 'race-explorer.json'
            with (
                patch('src.pipeline.VALIDATION_METRICS_JSON', str(validation_path)),
                patch('src.pipeline.RACE_EXPLORER_JSON', str(race_path)),
            ):
                _write_validation_metrics_json(pd.DataFrame(rows), config)
                _write_race_explorer_json({
                    (2024, 'Bahrain Grand Prix', 'VER'): {
                        'actualPitLaps': [4],
                        'laps': {
                            2: {
                                'lapNumber': 2,
                                'lapTime': 91.2,
                                'hmmStateProbabilities': [0.2, 0.8],
                                'hmmAlarm': True,
                                'tyreAgeAlarm': False,
                                'logisticAlarm': False,
                                'actualPit': False,
                            },
                        },
                    },
                }, 0.6, 0.8)

            validation = json.loads(validation_path.read_text(encoding='utf-8'))
            race_explorer = json.loads(race_path.read_text(encoding='utf-8'))

        self.assertEqual(validation['methods']['HMM']['nStates'], 3)
        self.assertEqual(
            [point['threshold'] for point in validation['methods']['HMM']['thresholds']],
            [0.3, 0.6],
        )
        self.assertEqual(race_explorer['year'], 2024)
        driver = race_explorer['races'][0]['drivers'][0]
        self.assertEqual(driver['actualPitLaps'], [4])
        self.assertEqual(driver['laps'][0]['hmmStateProbabilities'], [0.2, 0.8])

    def test_final_2024_evaluation_rejects_repeat_run(self):
        data = pd.DataFrame({
            'Year': [2023, FINAL_TEST_YEAR],
            'GrandPrix': ['Bahrain', 'Bahrain'],
            'Driver': ['VER', 'VER'],
        })
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'final_test.complete'
            marker.write_text('already evaluated', encoding='utf-8')
            with (
                patch('src.pipeline._load_data', return_value=data),
                patch('src.pipeline.FINAL_TEST_MARKER', str(marker)),
                self.assertRaisesRegex(RuntimeError, 'already been run'),
            ):
                run_evaluate(test_year=FINAL_TEST_YEAR)

    def test_stint_with_missing_compounds_uses_unknown_label(self):
        rows = [
            _lap(2023, 'Bahrain', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 4)
        ]
        rows[0]['Compound'] = None
        rows[1]['Compound'] = 'None'
        rows[2]['Compound'] = 'unknown'

        features = build_observation_matrix(pd.DataFrame(rows))
        records = get_stint_records(features)

        self.assertTrue(features.empty)
        self.assertEqual(records, [])

    def test_laps_with_missing_or_invalid_tyre_age_are_excluded(self):
        rows = [
            _lap(2023, 'Bahrain', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 6)
        ]
        rows[1]['TyreLife'] = None
        rows[3]['TyreLife'] = -1

        features = build_observation_matrix(pd.DataFrame(rows))

        self.assertEqual(features['LapNumber'].tolist(), [1, 3, 5])
        self.assertTrue(features['tyre_age_norm'].notna().all())

    def test_unmarked_out_lap_at_new_stint_start_is_removed(self):
        rows = [
            _lap(2023, 'Bahrain', 'HAM', lap, 1, 100.0 + lap)
            for lap in range(1, 4)
        ]
        rows.extend([
            _lap(2023, 'Bahrain', 'HAM', 4, 2, 120.0),
            _lap(2023, 'Bahrain', 'HAM', 5, 2, 101.0),
            _lap(2023, 'Bahrain', 'HAM', 6, 2, 102.0),
            _lap(2023, 'Bahrain', 'HAM', 7, 2, 103.0),
            _lap(2023, 'Bahrain', 'HAM', 8, 2, 104.0, pit_in=True),
        ])

        features = build_observation_matrix(pd.DataFrame(rows))
        second_stint = features[features['Stint'] == 2]

        self.assertEqual(second_stint['LapNumber'].tolist(), [5, 6, 7])
        self.assertTrue((second_stint['actual_pit_lap'] == 8).all())
        self.assertLess(second_stint['lap_time_delta'].max(), 3.0)

    def test_extreme_lap_time_outlier_is_excluded(self):
        rows = [
            _lap(2023, 'Bahrain', 'VER', 1, 1, 100.0),
            _lap(2023, 'Bahrain', 'VER', 2, 1, 150.0),
            _lap(2023, 'Bahrain', 'VER', 3, 1, 101.0),
            _lap(2023, 'Bahrain', 'VER', 4, 1, 102.0),
            _lap(2023, 'Bahrain', 'VER', 5, 1, 103.0),
        ]

        features = build_observation_matrix(pd.DataFrame(rows))

        self.assertNotIn(2, features['LapNumber'].tolist())

    def test_outlier_filter_uses_only_prior_laps(self):
        prefix = pd.DataFrame([
            _lap(2023, 'Bahrain', 'VER', lap, 1, lap_time)
            for lap, lap_time in enumerate([130.0, 100.0, 100.0, 100.0, 100.0], 1)
        ])
        future = pd.DataFrame([
            _lap(2023, 'Bahrain', 'VER', lap, 1, 80.0)
            for lap in range(6, 11)
        ])

        prefix_features = build_observation_matrix(prefix)
        extended_features = build_observation_matrix(pd.concat([prefix, future]))
        extended_prefix = extended_features[extended_features['LapNumber'] <= 5]

        self.assertEqual(prefix_features['LapNumber'].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(extended_prefix['LapNumber'].tolist(), [1, 2, 3, 4, 5])
        np.testing.assert_allclose(
            prefix_features[FEATURE_COLUMNS].to_numpy(),
            extended_prefix[FEATURE_COLUMNS].to_numpy(),
        )

    def test_cumulative_best_features_do_not_depend_on_future_laps(self):
        prefix = pd.DataFrame([
            _lap(2023, 'Bahrain', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 4)
        ])
        future = pd.DataFrame([_lap(2023, 'Bahrain', 'VER', 4, 1, 90.0)])

        prefix_features = build_observation_matrix(prefix)
        extended_features = build_observation_matrix(pd.concat([prefix, future]))

        np.testing.assert_allclose(
            prefix_features[FEATURE_COLUMNS].to_numpy(),
            extended_features.loc[:2, FEATURE_COLUMNS].to_numpy(),
        )

    def test_feature_generation_marks_current_feature_version(self):
        raw = pd.DataFrame([
            _lap(2023, 'Bahrain', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 4)
        ])

        features = build_observation_matrix(raw)

        self.assertTrue((features['FeatureVersion'] == 5).all())

    def test_loader_rejects_processed_data_without_current_feature_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'processed.csv'
            old_data = {column: [1] for column in (
                ['Year', 'GrandPrix', 'Driver', 'Stint', 'LapNumber']
                + FEATURE_COLUMNS
                + ['actual_pit_lap', 'Compound', 'TyreLife', 'FeatureVersion']
            )}
            old_data['GrandPrix'] = ['Bahrain']
            old_data['Driver'] = ['VER']
            old_data['Compound'] = ['SOFT']
            old_data['FeatureVersion'] = [2]
            pd.DataFrame(old_data).to_csv(path, index=False)
            with patch('src.pipeline.DATA_PATH', str(path)):
                with self.assertRaisesRegex(ValueError, 'outdated or inconsistent features'):
                    _load_data()

    def test_season_race_list_uses_full_non_testing_schedule(self):
        schedule = pd.DataFrame({
            'EventName': ['Bahrain Grand Prix', 'Australian Grand Prix'],
        })
        with patch('src.pipeline.fastf1.get_event_schedule', return_value=schedule) as get_schedule:
            races = _get_season_races([2024])

        self.assertEqual(
            races,
            [(2024, 'Bahrain Grand Prix'), (2024, 'Australian Grand Prix')],
        )
        get_schedule.assert_called_once_with(2024, include_testing=False)

    def test_preprocessing_excludes_entire_race_if_any_driver_used_wet_tyres(self):
        dry_laps = pd.DataFrame([
            _lap(2023, 'Dry Grand Prix', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 5)
        ])
        wet_laps = pd.DataFrame([
            _lap(2023, 'Wet Grand Prix', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 5)
        ])
        wet_laps.loc[wet_laps['LapNumber'].eq(3), 'Compound'] = 'Intermediate'

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'processed.csv'
            raw_path = Path(directory) / 'raw_laps.csv'
            with (
                patch('src.pipeline._get_season_races', return_value=[
                    (2023, 'Dry Grand Prix'),
                    (2023, 'Wet Grand Prix'),
                ]),
                patch(
                    'src.pipeline.load_race',
                    side_effect=[dry_laps, wet_laps],
                ),
                patch('src.pipeline.DATA_PATH', str(path)),
                patch('src.pipeline.RAW_LAPS_PATH', str(raw_path)),
            ):
                run_preprocess(years=[2023])

            processed = pd.read_csv(path)
            raw_laps = pd.read_csv(raw_path)

        self.assertEqual(processed['GrandPrix'].unique().tolist(), ['Dry Grand Prix'])
        self.assertEqual(raw_laps['GrandPrix'].unique().tolist(), ['Dry Grand Prix'])

    def test_preprocessing_resumes_from_saved_race_laps(self):
        first_race = pd.DataFrame([
            _lap(2023, 'First Grand Prix', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 5)
        ])
        second_race = pd.DataFrame([
            _lap(2023, 'Second Grand Prix', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 5)
        ])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'processed.csv'
            raw_path = Path(directory) / 'raw_laps.csv'
            with (
                patch('src.pipeline._get_season_races', return_value=[
                    (2023, 'First Grand Prix'),
                    (2023, 'Second Grand Prix'),
                ]),
                patch(
                    'src.pipeline.load_race',
                    side_effect=[
                        first_race,
                        RuntimeError('temporary fetch failure'),
                        second_race,
                    ],
                ) as load_race,
                patch('src.pipeline.DATA_PATH', str(path)),
                patch('src.pipeline.RAW_LAPS_PATH', str(raw_path)),
            ):
                run_preprocess(years=[2023])
                run_preprocess(years=[2023])

            processed = pd.read_csv(path)
            raw_laps = pd.read_csv(raw_path)

        self.assertEqual(load_race.call_count, 3)
        self.assertEqual(
            set(processed['GrandPrix'].unique()),
            {'First Grand Prix', 'Second Grand Prix'},
        )
        self.assertEqual(
            set(raw_laps['GrandPrix'].unique()),
            {'First Grand Prix', 'Second Grand Prix'},
        )

    def test_preprocessing_stops_on_rate_limit_and_saves_checkpoint(self):
        first_race = pd.DataFrame([
            _lap(2023, 'First Grand Prix', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 5)
        ])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'processed.csv'
            raw_path = Path(directory) / 'raw_laps.csv'
            with (
                patch('src.pipeline._get_season_races', return_value=[
                    (2023, 'First Grand Prix'),
                    (2023, 'Rate Limited Grand Prix'),
                    (2023, 'Not Attempted Grand Prix'),
                ]),
                patch(
                    'src.pipeline.load_race',
                    side_effect=[
                        first_race,
                        RateLimitExceededError('500 calls/h'),
                    ],
                ) as load_race,
                patch('src.pipeline.DATA_PATH', str(path)),
                patch('src.pipeline.RAW_LAPS_PATH', str(raw_path)),
            ):
                run_preprocess(years=[2023])

            processed = pd.read_csv(path)
            raw_laps = pd.read_csv(raw_path)

        self.assertEqual(load_race.call_count, 2)
        self.assertEqual(processed['GrandPrix'].unique().tolist(), ['First Grand Prix'])
        self.assertEqual(raw_laps['GrandPrix'].unique().tolist(), ['First Grand Prix'])


if __name__ == '__main__':
    unittest.main()
