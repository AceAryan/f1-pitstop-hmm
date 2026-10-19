"""Season-split preprocessing, training, and pit-stop evaluation."""

import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import fastf1
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_race
from src.features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    RACE_COLUMNS,
    STINT_COLUMNS,
    build_observation_matrix,
    get_stint_records,
    scale_features,
)
from src.hmm import GaussianHMM


DATA_PATH = 'data/processed.csv'
MODELS_DIR = 'results/models'
PLOTS_DIR = 'results/plots'
RESULTS_CSV = 'results/evaluation_summary.csv'
POOLED_RESULTS_CSV = 'results/evaluation_pooled_summary.csv'
TIMING_DELTAS_CSV = 'results/prediction_timing_deltas.csv'
TUNING_RESULTS_CSV = 'results/validation_tuning.csv'
TUNING_CONFIG_PATH = 'results/validation_tuning.pkl'
FINAL_TEST_MARKER = 'results/final_test_2024.complete'

DEFAULT_SEASONS = (2020, 2021, 2022, 2023)
COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')
PIT_LOOKAHEAD_LAPS = 3
TRAINING_SEASONS = (2020, 2021, 2022)
VALIDATION_YEAR = 2023
FINAL_TEST_YEAR = 2024
HMM_STATE_CANDIDATES = (2, 3, 4)
HMM_ALARM_THRESHOLDS = tuple(round(value / 10, 1) for value in range(3, 10))
TYRE_AGE_THRESHOLDS = tuple(round(value / 100, 2) for value in range(60, 100, 5))
DEFAULT_HMM_ALARM_THRESHOLD = 0.5
DEFAULT_TYRE_AGE_THRESHOLD = 0.8


def _get_season_races(years: list[int]) -> list[tuple[int, str]]:
    races = []
    for season in years:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        races.extend(
            (season, str(event_name))
            for event_name in schedule['EventName'].dropna()
        )
    return races


def run_preprocess(
    year: int | None = None,
    gp: list[str] | None = None,
    years: list[int] | None = None,
) -> None:
    if years and year:
        raise ValueError('Choose either --year or --years, not both.')
    if year and gp:
        races = [(year, name) for name in gp]
    elif years:
        races = _get_season_races(years)
    elif year:
        races = _get_season_races([year])
    else:
        races = _get_season_races(list(DEFAULT_SEASONS))

    print(f'Loading races: {races}')
    os.makedirs('data', exist_ok=True)
    frames = []
    loaded_races = []
    skipped_races = []
    wet_races = []
    for race_year, name in races:
        try:
            laps_df = load_race(race_year, name)
            compounds_used = set(
                laps_df['Compound'].str.upper().dropna().unique()
            )
            if compounds_used & {'INTERMEDIATE', 'WET'}:
                wet_races.append((race_year, name))
                print(
                    f'  Excluding {race_year} {name}: '
                    'intermediate or wet-compound tyres were used.'
                )
                continue
            frames.append(laps_df)
            loaded_races.append((race_year, name))
        except Exception as error:
            skipped_races.append((race_year, name))
            print(f'  Skipping {race_year} {name}: {error}')
    if not frames:
        if wet_races and not skipped_races:
            raise RuntimeError(
                'No dry races loaded; every successfully loaded race used '
                'intermediate or wet-compound tyres.'
            )
        raise RuntimeError(
            'No dry races loaded. Check the FastF1 cache, network access, and '
            'whether the requested races used dry-compound tyres.'
        )

    features = build_observation_matrix(pd.concat(frames, ignore_index=True))
    features.to_csv(DATA_PATH, index=False)
    loaded_by_year: dict[int, int] = defaultdict(int)
    for season, _ in loaded_races:
        loaded_by_year[season] += 1
    print(
        f'Saved {len(features)} clean lap observations from {len(loaded_races)} '
        f'of {len(races)} requested races to {DATA_PATH}.'
    )
    print(f'Loaded races by season: {dict(sorted(loaded_by_year.items()))}')
    if wet_races:
        print(f'Excluded {len(wet_races)} races with intermediate or wet tyres.')
    if skipped_races:
        print(
            f'WARNING: {len(skipped_races)} race sessions failed to load; '
            'the processed CSV and downstream evaluation cover only the '
            'successfully loaded sessions.'
        )


def _load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'{DATA_PATH} not found; run with --preprocess first.')
    data = pd.read_csv(DATA_PATH)
    required = set(RACE_COLUMNS + STINT_COLUMNS[3:] + FEATURE_COLUMNS + [
        'LapNumber', 'actual_pit_lap', 'Compound', 'TyreLife', 'FeatureVersion'
    ])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f'{DATA_PATH} is stale or incomplete (missing {missing}); '
            'regenerate it with --preprocess.'
        )
    versions = pd.to_numeric(data['FeatureVersion'], errors='coerce').dropna().unique()
    if len(versions) != 1 or versions[0] != FEATURE_VERSION:
        raise ValueError(
            f'{DATA_PATH} contains outdated or inconsistent features; '
            'regenerate it with --preprocess before training or evaluation.'
        )
    data['Year'] = pd.to_numeric(data['Year'], errors='raise').astype(int)
    data['LapNumber'] = pd.to_numeric(data['LapNumber'], errors='raise').astype(int)
    return data


def _resolve_test_year(data: pd.DataFrame, test_year: int | None) -> int:
    years = sorted(data['Year'].unique())
    resolved = int(test_year if test_year is not None else years[-1])
    if resolved not in years:
        raise ValueError(f'Test season {resolved} is not present in {DATA_PATH}; found {years}.')
    if not any(year < resolved for year in years):
        raise ValueError(
            f'No earlier training season exists before test season {resolved}; '
            f'available seasons are {years}.'
        )
    return resolved


def _feature_columns(include_tyre_age: bool = True) -> list[str]:
    return [
        column for column in FEATURE_COLUMNS
        if include_tyre_age or column != 'tyre_age_norm'
    ]


def _group_records(records: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[column] for column in RACE_COLUMNS)].append(record)
    return grouped


def _actual_pits_by_race(data: pd.DataFrame) -> dict[tuple, list[int]]:
    actual: dict[tuple, list[int]] = {}
    for key, race_driver in data.groupby(RACE_COLUMNS, sort=False):
        actual[key] = sorted(
            set(int(value) for value in race_driver['actual_pit_lap'].dropna().unique())
        )
    return actual


def _fit_hmm_models(
    train_data: pd.DataFrame,
    test_year: int,
    n_states: int,
    n_iter: int,
    compounds: list[str],
    include_tyre_age: bool,
) -> dict[str, dict]:
    if n_states < 2:
        raise ValueError('HMM must have at least 2 states.')
    records = get_stint_records(train_data)
    feature_columns = _feature_columns(include_tyre_age)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record['compound'] in compounds:
            grouped[record['compound']].append(record)

    bundles = {}
    for compound in compounds:
        compound_records = grouped.get(compound, [])
        if len(compound_records) < 5:
            print(
                f'[{compound}] {len(compound_records)} training stints; '
                'need at least 5, skipping.'
            )
            continue
        indices = [FEATURE_COLUMNS.index(column) for column in feature_columns]
        raw_sequences = [record['observations'][:, indices] for record in compound_records]
        scaled_sequences, scaler = scale_features(raw_sequences)
        model = GaussianHMM(n_states=n_states, n_iter=n_iter)
        model.fit(scaled_sequences)
        bundles[compound] = {
            'model': model,
            'scaler': scaler,
            'test_year': test_year,
            'train_years': sorted(train_data['Year'].unique().tolist()),
            'n_states': n_states,
            'feature_columns': feature_columns,
            'include_tyre_age': include_tyre_age,
        }
    return bundles


def _fit_baseline(train_data: pd.DataFrame):
    target_distance = train_data['actual_pit_lap'] - train_data['LapNumber']
    labels = target_distance.between(1, PIT_LOOKAHEAD_LAPS).astype(int)
    if labels.nunique() != 2:
        raise ValueError(
            'Training data must contain both laps within and outside the '
            f'{PIT_LOOKAHEAD_LAPS}-lap pit window to fit logistic regression.'
        )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    )
    classifier.fit(train_data[FEATURE_COLUMNS], labels)
    return classifier


def _save_model_bundles(bundles: dict[str, dict], model_name: str) -> list[str]:
    trained = []
    for compound, bundle in bundles.items():
        if model_name not in {'hmm', 'hmm_no_age'}:
            raise ValueError(f'Unknown HMM model variant: {model_name}')
        model_path = os.path.join(
            MODELS_DIR, f'{model_name}_{compound.lower()}.pkl'
        )
        with open(model_path, 'wb') as model_file:
            pickle.dump(bundle, model_file)
        trained.append(compound)
        print(f'[{model_name} {compound}] trained -> {model_path}')
    return trained


def run_tune(n_iter: int = 50, compounds: list[str] | None = None) -> dict:
    """Tune on 2023 only; all candidate models are fit on 2020–2022."""
    if os.path.exists(FINAL_TEST_MARKER):
        raise RuntimeError('The 2024 final test has already run; do not retune afterward.')
    data = _load_data()
    required_years = set(TRAINING_SEASONS) | {VALIDATION_YEAR}
    available_years = set(data['Year'].unique())
    if not required_years.issubset(available_years):
        raise ValueError(
            f'Validation tuning requires seasons {sorted(required_years)}; '
            f'found {sorted(available_years)}.'
        )
    train_data = data[data['Year'].isin(TRAINING_SEASONS)].copy()
    validation_data = data[data['Year'].eq(VALIDATION_YEAR)].copy()
    validation_groups = _group_records(get_stint_records(validation_data))
    actual_by_race = _actual_pits_by_race(validation_data)
    requested = [compound.upper() for compound in (compounds or COMPOUNDS)]
    tuning_rows = []

    for include_tyre_age, method in ((True, 'HMM'), (False, 'HMMNoAge')):
        for n_states in HMM_STATE_CANDIDATES:
            bundles = _fit_hmm_models(
                train_data, VALIDATION_YEAR, n_states, n_iter, requested, include_tyre_age
            )
            if not bundles:
                raise RuntimeError(f'No {method} models could be fit for validation tuning.')
            posterior_by_race: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
            for race_key, records in validation_groups.items():
                for record in records:
                    bundle = bundles.get(record['compound'])
                    if bundle is None:
                        continue
                    indices = [FEATURE_COLUMNS.index(col) for col in bundle['feature_columns']]
                    observations = record['observations'][:, indices]
                    posterior = bundle['model'].filter(
                        bundle['scaler'].transform(observations)
                    )[:, -1]
                    posterior_by_race[race_key].append((posterior, record['lap_numbers']))
            for threshold in HMM_ALARM_THRESHOLDS:
                predicted_by_race = {}
                for race_key, stints in posterior_by_race.items():
                    predictions = []
                    for posterior, laps in stints:
                        indices = np.flatnonzero(posterior >= threshold)
                        if indices.size:
                            predictions.append(int(laps[indices[0]]))
                    predicted_by_race[race_key] = predictions
                tuning_rows.append({
                    'Method': method,
                    'NStates': n_states,
                    'Threshold': threshold,
                    **_pooled_metrics(predicted_by_race, actual_by_race, tolerance=2),
                })

    for threshold in TYRE_AGE_THRESHOLDS:
        predicted_by_race: dict[tuple, list[int]] = defaultdict(list)
        for race_key, records in validation_groups.items():
            for record in records:
                age = record['observations'][:, FEATURE_COLUMNS.index('tyre_age_norm')]
                indices = np.flatnonzero(age >= threshold)
                if indices.size:
                    predicted_by_race[race_key].append(int(record['lap_numbers'][indices[0]]))
        tuning_rows.append({
            'Method': 'TyreAge',
            'NStates': np.nan,
            'Threshold': threshold,
            **_pooled_metrics(predicted_by_race, actual_by_race, tolerance=2),
        })

    logistic = _fit_baseline(train_data)
    logistic_predictions: dict[tuple, list[int]] = defaultdict(list)
    for race_key, records in validation_groups.items():
        for record in records:
            flags = logistic.predict(
                pd.DataFrame(record['observations'], columns=FEATURE_COLUMNS)
            ).astype(bool)
            logistic_predictions[race_key].extend(_boolean_entries(flags, record['lap_numbers']))
    tuning_rows.append({
        'Method': 'Logistic',
        'NStates': np.nan,
        'Threshold': np.nan,
        **_pooled_metrics(logistic_predictions, actual_by_race, tolerance=2),
    })

    tuning = pd.DataFrame(tuning_rows)
    os.makedirs(os.path.dirname(TUNING_RESULTS_CSV), exist_ok=True)
    tuning.to_csv(TUNING_RESULTS_CSV, index=False)
    selected = {}
    for method, prefix in (
        ('HMM', 'hmm'),
        ('HMMNoAge', 'hmm_no_age'),
        ('TyreAge', 'tyre_age'),
    ):
        candidates = tuning[tuning['Method'].eq(method)].sort_values(
            ['F1', 'Precision', 'Recall', 'NStates', 'Threshold'],
            ascending=[False, False, False, True, False],
            na_position='last',
        )
        best = candidates.iloc[0]
        if method == 'TyreAge':
            selected['tyre_age_threshold'] = float(best['Threshold'])
        else:
            selected[f'{prefix}_states'] = int(best['NStates'])
            selected[f'{prefix}_threshold'] = float(best['Threshold'])
        print(
            f'{method} selected: states={best["NStates"]}, '
            f'threshold={best["Threshold"]}, pooled F1={best["F1"]:.3f}'
        )

    config = {
        'validation_year': VALIDATION_YEAR,
        'training_years': list(TRAINING_SEASONS),
        'compounds': requested,
        'n_iter': n_iter,
        **selected,
    }
    with open(TUNING_CONFIG_PATH, 'wb') as config_file:
        pickle.dump(config, config_file)
    print(f'Saved 2023 tuning results to {TUNING_RESULTS_CSV}')
    print(f'Saved selected configuration to {TUNING_CONFIG_PATH}')
    return config


def run_train(
    test_year: int | None = None,
    compounds: list[str] | None = None,
    n_iter: int = 50,
    n_states: int = 3,
) -> None:
    data = _load_data()
    test_year = _resolve_test_year(data, test_year)
    requested = [compound.upper() for compound in (compounds or COMPOUNDS)]
    train_data = data[data['Year'] < test_year].copy()
    tuned_config = None
    final_fit = test_year == FINAL_TEST_YEAR
    if final_fit:
        if os.path.exists(FINAL_TEST_MARKER):
            raise RuntimeError(
                f'The final {FINAL_TEST_YEAR} evaluation is already locked; '
                'do not replace its trained models.'
            )
        if not os.path.exists(TUNING_CONFIG_PATH):
            raise FileNotFoundError(
                f'{TUNING_CONFIG_PATH} not found; run --tune on 2023 before final training.'
            )
        with open(TUNING_CONFIG_PATH, 'rb') as config_file:
            tuned_config = pickle.load(config_file)
        if tuned_config.get('validation_year') != VALIDATION_YEAR:
            raise ValueError('Tuning configuration is not for the required 2023 validation season.')
        if set(tuned_config.get('training_years', [])) != set(TRAINING_SEASONS):
            raise ValueError('Tuning configuration was not fit on exactly 2020–2022.')
        if set(tuned_config.get('compounds', [])) != set(requested):
            raise ValueError(
                'Final model compounds differ from the compounds used during 2023 tuning.'
            )
        full_states = int(tuned_config['hmm_states'])
        no_age_states = int(tuned_config['hmm_no_age_states'])
        n_iter = int(tuned_config['n_iter'])
        final_training_years = TRAINING_SEASONS + (VALIDATION_YEAR,)
        train_data = data[data['Year'].isin(final_training_years)].copy()
        train_years = set(train_data['Year'].unique())
        if set(final_training_years) != train_years:
            raise ValueError('Final 2024 training requires seasons 2020 through 2023.')
    else:
        if n_states < 2:
            raise ValueError('HMM must have at least 2 states.')
        full_states = no_age_states = n_states

    train_records = get_stint_records(train_data)
    if not train_records:
        raise ValueError(f'No complete training stints found before {test_year}.')
    print(
        f'Training seasons: {sorted(train_data["Year"].unique().tolist())}; '
        f'held-out season: {test_year}; training stints: {len(train_records)}'
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    full_bundles = _fit_hmm_models(
        train_data, test_year, full_states, n_iter, requested, include_tyre_age=True
    )
    no_age_bundles = _fit_hmm_models(
        train_data, test_year, no_age_states, n_iter, requested, include_tyre_age=False
    )
    trained = _save_model_bundles(full_bundles, 'hmm')
    no_age_trained = _save_model_bundles(no_age_bundles, 'hmm_no_age')
    if not trained or not no_age_trained:
        raise RuntimeError('At least one full and one no-tyre-age HMM must train successfully.')

    baseline = _fit_baseline(train_data)
    baseline_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    with open(baseline_path, 'wb') as model_file:
        pickle.dump({
            'model': baseline,
            'test_year': test_year,
            'train_years': sorted(train_data['Year'].unique().tolist()),
            'feature_columns': FEATURE_COLUMNS,
        }, model_file)
    metadata = {
        'test_year': test_year,
        'compounds': trained,
        'no_age_compounds': no_age_trained,
        'train_years': sorted(train_data['Year'].unique().tolist()),
        'final_fit': final_fit,
        'feature_version': FEATURE_VERSION,
        'tuned_config': tuned_config,
    }
    with open(os.path.join(MODELS_DIR, 'training_metadata.pkl'), 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)
    if tuned_config:
        print(
            f'Final tuning settings: HMM states={full_states}, '
            f'HMM alarm={tuned_config["hmm_threshold"]:.1f}, '
            f'ablation states={no_age_states}, '
            f'ablation alarm={tuned_config["hmm_no_age_threshold"]:.1f}, '
            f'tyre-age threshold={tuned_config["tyre_age_threshold"]:.2f}'
        )
    print(f'Logistic baseline trained on seasons < {test_year} -> {baseline_path}')


def evaluate_predictions(predicted: list[int], actual: list[int], tolerance: int) -> dict:
    pairs = _match_laps(predicted, actual, tolerance)
    true_positives = len(pairs)
    false_positives = len(predicted) - true_positives
    false_negatives = len(actual) - true_positives
    precision = true_positives / (true_positives + false_positives) if predicted else 0.0
    recall = true_positives / (true_positives + false_negatives) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3),
    }


def _match_laps(
    predicted: list[int],
    actual: list[int],
    tolerance: int | None,
) -> list[tuple[int, int]]:
    """Maximize one-to-one matches, preferring the smallest lap difference."""
    if not predicted or not actual:
        return []
    predictions = np.asarray(predicted, dtype=int)
    pits = np.asarray(actual, dtype=int)
    distances = np.abs(predictions[:, None] - pits[None, :])
    max_distance = int(distances.max())
    max_matches = min(len(predicted), len(actual))
    unmatched_cost = (max_distance + 1) * (max_matches + 1)
    invalid_cost = unmatched_cost * 2
    costs = np.full((len(predictions), len(pits) + len(predictions)), unmatched_cost, dtype=float)
    actual_costs = costs[:, :len(pits)]
    actual_costs[:] = distances
    if tolerance is not None:
        actual_costs[distances > tolerance] = invalid_cost
    rows, columns = linear_sum_assignment(costs)
    return [
        (int(predictions[row]), int(pits[column]))
        for row, column in zip(rows, columns)
        if column < len(pits)
        and (tolerance is None or distances[row, column] <= tolerance)
    ]


def _pooled_metrics(
    predictions_by_race: dict[tuple, list[int]],
    actual_by_race: dict[tuple, list[int]],
    tolerance: int,
) -> dict:
    true_positives = false_positives = false_negatives = 0
    for race_key in sorted(set(predictions_by_race) | set(actual_by_race)):
        predicted = predictions_by_race.get(race_key, [])
        actual = actual_by_race.get(race_key, [])
        matches = _match_laps(predicted, actual, tolerance)
        true_positives += len(matches)
        false_positives += len(predicted) - len(matches)
        false_negatives += len(actual) - len(matches)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3),
    }


def _first_threshold_entry(values: np.ndarray, lap_numbers: np.ndarray, threshold: float) -> list[int]:
    indices = np.flatnonzero(values >= threshold)
    return [int(lap_numbers[indices[0]])] if indices.size else []


def _timing_rows(
    method: str,
    race_key: tuple,
    predicted: list[int],
    actual: list[int],
) -> list[dict]:
    pairs = _match_laps(predicted, actual, tolerance=None)
    unmatched_predictions = list(predicted)
    unmatched_actuals = list(actual)
    rows = []
    for prediction, pit_lap in pairs:
        unmatched_predictions.remove(prediction)
        unmatched_actuals.remove(pit_lap)
        rows.append({
            **dict(zip(RACE_COLUMNS, race_key)),
            'Method': method,
            'PredictionLap': prediction,
            'ActualPitLap': pit_lap,
            'PredictedMinusActual': prediction - pit_lap,
            'MatchStatus': 'matched',
        })
    for prediction in unmatched_predictions:
        rows.append({
            **dict(zip(RACE_COLUMNS, race_key)),
            'Method': method,
            'PredictionLap': prediction,
            'ActualPitLap': np.nan,
            'PredictedMinusActual': np.nan,
            'MatchStatus': 'unmatched_prediction',
        })
    for pit_lap in unmatched_actuals:
        rows.append({
            **dict(zip(RACE_COLUMNS, race_key)),
            'Method': method,
            'PredictionLap': np.nan,
            'ActualPitLap': pit_lap,
            'PredictedMinusActual': np.nan,
            'MatchStatus': 'unmatched_actual',
        })
    return rows


def _boolean_entries(predicted: np.ndarray, lap_numbers: np.ndarray) -> list[int]:
    entries = []
    was_positive = False
    for flag, lap_number in zip(predicted, lap_numbers):
        if flag and not was_positive:
            entries.append(int(lap_number))
        was_positive = bool(flag)
    return entries


def _load_models(test_year: int) -> tuple[dict[str, dict[str, dict]], object, dict]:
    metadata_path = os.path.join(MODELS_DIR, 'training_metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'{metadata_path} not found; run with --train first.')
    with open(metadata_path, 'rb') as metadata_file:
        metadata = pickle.load(metadata_file)
    if metadata.get('feature_version') != FEATURE_VERSION:
        raise ValueError(
            f'Model metadata uses feature version {metadata.get("feature_version")}; '
            f'retrain with feature version {FEATURE_VERSION}.'
        )
    if metadata.get('test_year') != test_year:
        raise ValueError(
            f'Model bundle was trained for test season {metadata.get("test_year")}; '
            f'retrain with --test_year {test_year}.'
        )

    models = {'HMM': {}, 'HMMNoAge': {}}
    model_variants = (
        ('HMM', metadata.get('compounds', []), 'hmm_{}.pkl'),
        ('HMMNoAge', metadata.get('no_age_compounds', []), 'hmm_no_age_{}.pkl'),
    )
    for method, compounds, filename in model_variants:
        for compound in compounds:
            path = os.path.join(MODELS_DIR, filename.format(compound.lower()))
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} is missing; retrain for test season {test_year}.')
            with open(path, 'rb') as model_file:
                bundle = pickle.load(model_file)
            if bundle.get('test_year') != test_year:
                raise ValueError(
                    f'{path} was trained for test season {bundle.get("test_year")}; '
                    f'retrain with --test_year {test_year}.'
                )
            models[method][compound] = bundle

    baseline_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f'{baseline_path} not found; run with --train first.')
    with open(baseline_path, 'rb') as model_file:
        baseline_bundle = pickle.load(model_file)
    if baseline_bundle.get('test_year') != test_year:
        raise ValueError(
            f'{baseline_path} was trained for test season '
            f'{baseline_bundle.get("test_year")}; retrain with --test_year {test_year}.'
        )
    if not models:
        raise RuntimeError(f'No trained HMMs found in {MODELS_DIR}.')
    if not models['HMM'] or not models['HMMNoAge']:
        raise RuntimeError(f'Full and no-age HMMs are required in {MODELS_DIR}.')
    return models, baseline_bundle['model'], metadata


def _save_state_plot(
    race_driver: pd.DataFrame,
    predictions: dict[str, list[int]],
    states_by_lap: dict[int, int],
    path: str,
) -> None:
    ordered = race_driver.sort_values('LapNumber')
    figure, (state_axis, time_axis) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for _, lap in ordered.iterrows():
        lap_number = int(lap['LapNumber'])
        if lap_number in states_by_lap:
            color = colors[min(states_by_lap[lap_number], len(colors) - 1)]
            state_axis.bar(lap_number, 1, color=color, width=0.8)
    state_axis.set_ylabel('Filtered state')
    state_axis.set_yticks([])

    time_axis.plot(ordered['LapNumber'], ordered['LapTime'], color='#3498db', marker='o')
    time_axis.set(xlabel='Lap number', ylabel='Lap time (s)')
    for method, color in (
        ('HMM', 'purple'),
        ('HMMNoAge', 'magenta'),
        ('TyreAge', 'orange'),
        ('Logistic', 'green'),
    ):
        for lap_number in predictions[method]:
            state_axis.axvline(lap_number, color=color, linestyle=':', alpha=0.8)
            time_axis.axvline(lap_number, color=color, linestyle=':', alpha=0.6)
    for actual in ordered['actual_pit_lap'].dropna().unique():
        state_axis.axvline(actual, color='blue', linestyle='--', alpha=0.6)
        time_axis.axvline(actual, color='blue', linestyle='--', alpha=0.6)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(figure)


def run_evaluate(
    test_year: int | None = None,
    tolerance: int = 2,
    drivers: list[str] | None = None,
    plot_drivers: list[str] | None = None,
) -> pd.DataFrame:
    data = _load_data()
    test_year = _resolve_test_year(data, test_year)
    if test_year == FINAL_TEST_YEAR and os.path.exists(FINAL_TEST_MARKER):
        raise RuntimeError(
            f'The final {FINAL_TEST_YEAR} evaluation has already been run; '
            'the one-time test is locked.'
        )
    if test_year == FINAL_TEST_YEAR and (drivers is not None or tolerance != 2):
        raise ValueError(
            'The one-time 2024 test must include all drivers and use the '
            'predeclared +/-2 primary tolerance; +/-1 and +/-3 are reported automatically.'
        )
    test_data = data[data['Year'] == test_year].copy()
    test_races = sorted(test_data['GrandPrix'].unique().tolist())
    print(
        f'Evaluating season {test_year}: {len(test_races)} loaded races, '
        f'{test_data["Driver"].nunique()} drivers, '
        f'{test_data.groupby(RACE_COLUMNS).ngroups} race/driver entries.'
    )
    print(f'Test races: {test_races}')
    models, logistic, metadata = _load_models(test_year)
    if test_year == FINAL_TEST_YEAR and not metadata.get('final_fit'):
        raise ValueError(
            '2024 final evaluation requires models refit on 2020–2023 '
            'using the saved 2023 tuning configuration.'
        )
    config = metadata.get('tuned_config') or {}
    hmm_threshold = float(config.get('hmm_threshold', DEFAULT_HMM_ALARM_THRESHOLD))
    no_age_threshold = float(
        config.get('hmm_no_age_threshold', DEFAULT_HMM_ALARM_THRESHOLD)
    )
    tyre_age_threshold = float(
        config.get('tyre_age_threshold', DEFAULT_TYRE_AGE_THRESHOLD)
    )
    records = get_stint_records(test_data)
    selected_drivers = set(drivers) if drivers else set(test_data['Driver'].unique())
    plot_selection = set(plot_drivers or list(selected_drivers)[:3])
    records = [record for record in records if record['Driver'] in selected_drivers]

    grouped_records = _group_records(records)
    actual_by_race = {
        race_key: pits
        for race_key, pits in _actual_pits_by_race(test_data).items()
        if race_key[2] in selected_drivers
    }

    results = []
    predictions_by_method: dict[str, dict[tuple, list[int]]] = {
        method: {} for method in ('HMM', 'HMMNoAge', 'TyreAge', 'Logistic')
    }
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for race_key, race_driver_records in grouped_records.items():
        race_mask = pd.Series(True, index=test_data.index)
        for column, value in zip(RACE_COLUMNS, race_key):
            race_mask &= test_data[column].eq(value)
        race_driver = test_data.loc[race_mask]
        actual = actual_by_race.get(race_key, [])
        predictions: dict[str, list[int]] = {
            'HMM': [],
            'HMMNoAge': [],
            'TyreAge': [],
            'Logistic': [],
        }
        state_trace = {}

        for record in race_driver_records:
            observations = record['observations']
            laps = record['lap_numbers']
            compound = record['compound']

            logistic_flags = logistic.predict(
                pd.DataFrame(observations, columns=FEATURE_COLUMNS)
            ).astype(bool)
            predictions['Logistic'].extend(_boolean_entries(logistic_flags, laps))

            age_values = observations[:, FEATURE_COLUMNS.index('tyre_age_norm')]
            predictions['TyreAge'].extend(
                _first_threshold_entry(age_values, laps, tyre_age_threshold)
            )

            for method, threshold in (
                ('HMM', hmm_threshold),
                ('HMMNoAge', no_age_threshold),
            ):
                bundle = models[method].get(compound)
                if bundle is None:
                    continue
                indices = [FEATURE_COLUMNS.index(col) for col in bundle['feature_columns']]
                hmm_observations = observations[:, indices]
                filtered = bundle['model'].filter(
                    bundle['scaler'].transform(hmm_observations)
                )
                predictions[method].extend(
                    _first_threshold_entry(filtered[:, -1], laps, threshold)
                )
                if method == 'HMM':
                    states = filtered.argmax(axis=1)
                    state_trace.update(
                        {int(lap): int(state) for lap, state in zip(laps, states)}
                    )

        row = dict(zip(RACE_COLUMNS, race_key))
        row['ActualPitLaps'] = ';'.join(map(str, actual))
        for method, method_predictions in predictions.items():
            predictions_by_method[method][race_key] = sorted(method_predictions)
        for method, method_predictions in predictions.items():
            row[f'{method}PredictedPitLaps'] = ';'.join(map(str, sorted(method_predictions)))
            metrics = evaluate_predictions(method_predictions, actual, tolerance)
            for name, value in metrics.items():
                row[f'{method}_{name}'] = value
        results.append(row)

        if race_key[2] in plot_selection and state_trace:
            plot_name = f'states_{race_key[0]}_{race_key[1]}_{race_key[2]}'.replace(' ', '_')
            _save_state_plot(
                race_driver,
                predictions,
                state_trace,
                os.path.join(PLOTS_DIR, f'{plot_name}.png'),
            )

    if not results:
        raise ValueError(f'No complete evaluation stints found for test season {test_year}.')

    results_df = pd.DataFrame(results).sort_values(RACE_COLUMNS).reset_index(drop=True)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    timing_rows = []
    pooled_rows = []
    for method, race_predictions in predictions_by_method.items():
        race_keys = sorted(set(race_predictions) | set(actual_by_race))
        for race_key in race_keys:
            timing_rows.extend(
                _timing_rows(
                    method,
                    race_key,
                    race_predictions.get(race_key, []),
                    actual_by_race.get(race_key, []),
                )
            )
        for sensitivity_tolerance in (1, 2, 3):
            pooled_rows.append({
                'Year': test_year,
                'Method': method,
                'Tolerance': sensitivity_tolerance,
                'RaceDriverCount': len(race_keys),
                **_pooled_metrics(
                    race_predictions, actual_by_race, tolerance=sensitivity_tolerance
                ),
            })
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(POOLED_RESULTS_CSV, index=False)
    pd.DataFrame(timing_rows).to_csv(TIMING_DELTAS_CSV, index=False)
    print(f'Saved per-driver-race results to {RESULTS_CSV}')
    print(f'Saved pooled metrics and tolerance sensitivity to {POOLED_RESULTS_CSV}')
    print(f'Saved one-to-one prediction timing deltas to {TIMING_DELTAS_CSV}')
    for method, race_predictions in predictions_by_method.items():
        metrics = _pooled_metrics(race_predictions, actual_by_race, tolerance=tolerance)
        sensitivity = ', '.join(
            f'+/-{tol}: {_pooled_metrics(race_predictions, actual_by_race, tol)["F1"]:.3f}'
            for tol in (1, 2, 3)
        )
        print(
            f'{method} pooled P/R/F1 at +/-{tolerance}: '
            f'{metrics["Precision"]:.3f}/{metrics["Recall"]:.3f}/{metrics["F1"]:.3f} '
            f'({sensitivity})'
        )
        method_deltas = [
            row['PredictedMinusActual']
            for row in timing_rows
            if row['Method'] == method and row['MatchStatus'] == 'matched'
        ]
        unmatched_predictions = sum(
            row['Method'] == method and row['MatchStatus'] == 'unmatched_prediction'
            for row in timing_rows
        )
        unmatched_actuals = sum(
            row['Method'] == method and row['MatchStatus'] == 'unmatched_actual'
            for row in timing_rows
        )
        if method_deltas:
            delta_series = pd.Series(method_deltas)
            print(
                f'{method} prediction-minus-actual laps: matched={len(method_deltas)}, '
                f'median={delta_series.median():.1f}, '
                f'IQR=[{delta_series.quantile(0.25):.1f}, '
                f'{delta_series.quantile(0.75):.1f}], '
                f'unmatched predictions={unmatched_predictions}, '
                f'unmatched actual pits={unmatched_actuals}'
            )
        else:
            print(
                f'{method} prediction-minus-actual laps: no matched pairs, '
                f'unmatched predictions={unmatched_predictions}, '
                f'unmatched actual pits={unmatched_actuals}'
            )

    for method, bundles in models.items():
        for compound, bundle in bundles.items():
            model = bundle['model']
            labels = [f'State {index}' for index in range(model.n_states)]
            figure, axis = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                model.A,
                annot=True,
                fmt='.3f',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                vmin=0,
                vmax=1,
                ax=axis,
            )
            axis.set(
                title=f'{method} {compound} transition matrix',
                xlabel='To state',
                ylabel='From state',
            )
            figure.tight_layout()
            figure.savefig(
                os.path.join(PLOTS_DIR, f'transition_{method.lower()}_{compound.lower()}.png'),
                dpi=140,
            )
            plt.close(figure)
    if test_year == FINAL_TEST_YEAR:
        with open(FINAL_TEST_MARKER, 'x', encoding='utf-8') as marker:
            marker.write(f'Final test season {FINAL_TEST_YEAR} evaluated once.\n')
    return results_df
