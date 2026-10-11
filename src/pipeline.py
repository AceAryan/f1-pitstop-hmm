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

DEFAULT_SEASONS = (2020, 2021, 2022, 2023, 2024)
COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')
PIT_LOOKAHEAD_LAPS = 3
TYRE_AGE_PIT_THRESHOLD = 0.8


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
    for race_year, name in races:
        try:
            frames.append(load_race(race_year, name))
            loaded_races.append((race_year, name))
        except Exception as error:
            skipped_races.append((race_year, name))
            print(f'  Skipping {race_year} {name}: {error}')
    if not frames:
        raise RuntimeError('No races loaded. Check the FastF1 cache and network access.')

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


def _train_baseline(train_data: pd.DataFrame):
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


def run_train(
    test_year: int | None = None,
    compounds: list[str] | None = None,
    n_iter: int = 50,
    n_states: int = 3,
) -> None:
    if n_states != 3:
        raise ValueError('The ordered tyre-state model requires exactly 3 states.')
    data = _load_data()
    test_year = _resolve_test_year(data, test_year)
    train_data = data[data['Year'] < test_year].copy()
    train_records = get_stint_records(train_data)
    if not train_records:
        raise ValueError(f'No complete training stints found before {test_year}.')
    print(
        f'Training seasons: {sorted(train_data["Year"].unique().tolist())}; '
        f'held-out season: {test_year}; training stints: {len(train_records)}'
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    requested = [compound.upper() for compound in (compounds or COMPOUNDS)]
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in train_records:
        if record['compound'] in requested:
            grouped[record['compound']].append(record['observations'])

    trained = []
    for compound in requested:
        sequences = grouped.get(compound, [])
        if len(sequences) < 5:
            print(f'[{compound}] {len(sequences)} training stints; need at least 5, skipping.')
            continue
        scaled, scaler = scale_features(sequences)
        model = GaussianHMM(n_states=n_states, n_iter=n_iter)
        model.fit(scaled)
        model_path = os.path.join(MODELS_DIR, f'hmm_{compound.lower()}.pkl')
        with open(model_path, 'wb') as model_file:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'test_year': test_year,
                'train_years': sorted(train_data['Year'].unique().tolist()),
            }, model_file)
        trained.append(compound)
        print(f'[{compound}] trained on {len(sequences)} stints -> {model_path}')

    if not trained:
        raise RuntimeError('No compound HMMs were trained; check training data and compounds.')

    baseline = _train_baseline(train_data)
    baseline_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    with open(baseline_path, 'wb') as model_file:
        pickle.dump({
            'model': baseline,
            'test_year': test_year,
            'train_years': sorted(train_data['Year'].unique().tolist()),
        }, model_file)
    with open(os.path.join(MODELS_DIR, 'training_metadata.pkl'), 'wb') as metadata_file:
        pickle.dump({
            'test_year': test_year,
            'compounds': trained,
            'train_years': sorted(train_data['Year'].unique().tolist()),
        }, metadata_file)
    print(f'Logistic baseline trained on seasons < {test_year} -> {baseline_path}')


def evaluate_predictions(predicted: list[int], actual: list[int], tolerance: int) -> dict:
    matched: set[int] = set()
    true_positives = 0
    for prediction in sorted(predicted):
        candidates = [
            (abs(prediction - pit_lap), index)
            for index, pit_lap in enumerate(actual)
            if index not in matched and abs(prediction - pit_lap) <= tolerance
        ]
        if candidates:
            _, index = min(candidates)
            matched.add(index)
            true_positives += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(actual) - len(matched)
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


def _pit_entries(states: np.ndarray, lap_numbers: np.ndarray) -> list[int]:
    entries = []
    for index, state in enumerate(states):
        if state == 2 and (index == 0 or states[index - 1] != 2):
            entries.append(int(lap_numbers[index]))
    return entries


def _boolean_entries(predicted: np.ndarray, lap_numbers: np.ndarray) -> list[int]:
    entries = []
    was_positive = False
    for flag, lap_number in zip(predicted, lap_numbers):
        if flag and not was_positive:
            entries.append(int(lap_number))
        was_positive = bool(flag)
    return entries


def _load_models(test_year: int) -> tuple[dict, object]:
    metadata_path = os.path.join(MODELS_DIR, 'training_metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'{metadata_path} not found; run with --train first.')
    with open(metadata_path, 'rb') as metadata_file:
        metadata = pickle.load(metadata_file)
    if metadata.get('test_year') != test_year:
        raise ValueError(
            f'Model bundle was trained for test season {metadata.get("test_year")}; '
            f'retrain with --test_year {test_year}.'
        )

    models = {}
    for compound in metadata.get('compounds', []):
        path = os.path.join(MODELS_DIR, f'hmm_{compound.lower()}.pkl')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as model_file:
            bundle = pickle.load(model_file)
        if bundle.get('test_year') != test_year:
            raise ValueError(
                f'{path} was trained for test season {bundle.get("test_year")}; '
                f'retrain with --test_year {test_year}.'
            )
        models[compound] = bundle

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
    return models, baseline_bundle['model']


def _save_state_plot(
    race_driver: pd.DataFrame,
    predictions: dict[str, list[int]],
    states_by_lap: dict[int, int],
    path: str,
) -> None:
    ordered = race_driver.sort_values('LapNumber')
    figure, (state_axis, time_axis) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    for _, lap in ordered.iterrows():
        lap_number = int(lap['LapNumber'])
        if lap_number in states_by_lap:
            state_axis.bar(lap_number, 1, color=colors[states_by_lap[lap_number]], width=0.8)
    state_axis.set_ylabel('Filtered state')
    state_axis.set_yticks([])

    time_axis.plot(ordered['LapNumber'], ordered['LapTime'], color='#3498db', marker='o')
    time_axis.set(xlabel='Lap number', ylabel='Lap time (s)')
    for method, color in [('HMM', 'purple'), ('TyreAge', 'orange'), ('Logistic', 'green')]:
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
    test_data = data[data['Year'] == test_year].copy()
    test_races = sorted(test_data['GrandPrix'].unique().tolist())
    print(
        f'Evaluating season {test_year}: {len(test_races)} loaded races, '
        f'{test_data["Driver"].nunique()} drivers, '
        f'{test_data.groupby(RACE_COLUMNS).ngroups} race/driver entries.'
    )
    print(f'Test races: {test_races}')
    models, logistic = _load_models(test_year)
    records = get_stint_records(test_data)
    selected_drivers = set(drivers) if drivers else set(test_data['Driver'].unique())
    plot_selection = set(plot_drivers or list(selected_drivers)[:3])
    records = [record for record in records if record['Driver'] in selected_drivers]

    grouped_records: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[tuple(record[column] for column in RACE_COLUMNS)].append(record)

    results = []
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for race_key, race_driver_records in grouped_records.items():
        race_mask = pd.Series(True, index=test_data.index)
        for column, value in zip(RACE_COLUMNS, race_key):
            race_mask &= test_data[column].eq(value)
        race_driver = test_data.loc[race_mask]
        actual = sorted(
            set(int(value) for value in race_driver['actual_pit_lap'].dropna().unique())
        )
        predictions: dict[str, list[int]] = {
            'HMM': [],
            'TyreAge': [],
            'Logistic': [],
        }
        state_trace = {}

        for record in race_driver_records:
            observations = record['observations']
            laps = record['lap_numbers']
            compound = record['compound']

            logistic_flags = logistic.predict(observations).astype(bool)
            predictions['Logistic'].extend(_boolean_entries(logistic_flags, laps))

            age_flags = observations[:, FEATURE_COLUMNS.index('tyre_age_norm')] >= TYRE_AGE_PIT_THRESHOLD
            predictions['TyreAge'].extend(_boolean_entries(age_flags, laps))

            if compound not in models:
                continue
            bundle = models[compound]
            scaled = bundle['scaler'].transform(observations)
            filtered = bundle['model'].filter(scaled)
            states = filtered.argmax(axis=1)
            predictions['HMM'].extend(_pit_entries(states, laps))
            state_trace.update({int(lap): int(state) for lap, state in zip(laps, states)})

        row = dict(zip(RACE_COLUMNS, race_key))
        row['ActualPitLaps'] = ';'.join(map(str, actual))
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
    print(f'Saved season {test_year} comparison results to {RESULTS_CSV}')
    for method in ('HMM', 'TyreAge', 'Logistic'):
        print(f'{method} mean F1: {results_df[f"{method}_F1"].mean():.3f}')

    for compound, bundle in models.items():
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            bundle['model'].A,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=['FRESH', 'DEGRADING', 'CRITICAL'],
            yticklabels=['FRESH', 'DEGRADING', 'CRITICAL'],
            vmin=0,
            vmax=1,
            ax=axis,
        )
        axis.set(title=f'{compound} transition matrix', xlabel='To state', ylabel='From state')
        figure.tight_layout()
        figure.savefig(os.path.join(PLOTS_DIR, f'transition_{compound.lower()}.png'), dpi=140)
        plt.close(figure)
    return results_df
