import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_observation_matrix, get_stint_records
from src.hmm import GaussianHMM
from src.pipeline import _get_season_races, _load_data, evaluate_predictions


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

    def test_pit_matching_uses_real_lap_numbers(self):
        metrics = evaluate_predictions([72], [72], tolerance=2)
        self.assertEqual(metrics['TP'], 1)
        self.assertEqual(metrics['FP'], 0)
        self.assertEqual(metrics['FN'], 0)

    def test_stint_with_missing_compounds_uses_unknown_label(self):
        rows = [
            _lap(2023, 'Bahrain', 'VER', lap, 1, 100.0 + lap)
            for lap in range(1, 4)
        ]
        for row in rows:
            row['Compound'] = None

        features = build_observation_matrix(pd.DataFrame(rows))
        records = get_stint_records(features)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['compound'], 'UNKNOWN')

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

        self.assertTrue((features['FeatureVersion'] == 3).all())

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


if __name__ == '__main__':
    unittest.main()
