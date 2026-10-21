"""Feature engineering and race/stint sequence construction."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    'lap_time_delta',
    'rolling_degradation',
    'tyre_age_norm',
]
FEATURE_VERSION = 5
MAX_LAP_TIME_RATIO = 1.35
RACE_COLUMNS = ['Year', 'GrandPrix', 'Driver']
STINT_COLUMNS = RACE_COLUMNS + ['Stint']


def _is_clean_lap(df: pd.DataFrame) -> pd.Series:
    """Return laps suitable as model observations, excluding pit and non-green laps."""
    accurate = df['IsAccurate'].fillna(False).astype(bool)
    green = df['TrackStatus'].astype('string').eq('1').fillna(False)
    no_pit_activity = df['PitInTime'].isna() & df['PitOutTime'].isna()
    valid_lap_time = df['LapTime'].notna() & (df['LapTime'] > 0)
    first_lap = df['LapNumber'].ne(1) if 'RaceId' in df.columns else True
    return (
        accurate
        & green
        & no_pit_activity
        & valid_lap_time
        & df['Stint'].notna()
        & first_lap
    )


def _actual_pits_by_stint(df: pd.DataFrame, eligible: pd.Series) -> pd.DataFrame:
    """Associate each pit-in LapNumber with the last eligible stint observation before it."""
    eligible_laps = df.loc[eligible, STINT_COLUMNS + ['LapNumber']]
    pit_rows = df.loc[df['PitInTime'].notna(), RACE_COLUMNS + ['LapNumber']]
    pit_events = []

    for race_key, events in pit_rows.groupby(RACE_COLUMNS, sort=False, dropna=False):
        if not isinstance(race_key, tuple):
            race_key = (race_key,)
        race_mask = pd.Series(True, index=eligible_laps.index)
        for column, value in zip(RACE_COLUMNS, race_key):
            race_mask &= eligible_laps[column].eq(value)
        race_laps = eligible_laps.loc[race_mask].sort_values('LapNumber')

        for event_lap in events['LapNumber'].dropna():
            preceding = race_laps[race_laps['LapNumber'] < event_lap]
            if preceding.empty:
                continue
            last_lap = preceding.iloc[-1]
            pit_events.append({
                **{column: last_lap[column] for column in RACE_COLUMNS},
                'Stint': last_lap['Stint'],
                'actual_pit_lap': event_lap,
            })

    if not pit_events:
        return pd.DataFrame(columns=STINT_COLUMNS + ['actual_pit_lap'])

    return (
        pd.DataFrame(pit_events)
        .groupby(STINT_COLUMNS, as_index=False, dropna=False)['actual_pit_lap']
        .min()
    )


def build_observation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean lap observations while retaining real race and pit-lap metadata."""
    required = set(
        RACE_COLUMNS
        + ['Stint', 'LapNumber', 'LapTime', 'Compound', 'TyreLife',
           'PitInTime', 'PitOutTime', 'TrackStatus', 'IsAccurate']
    )
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f'Lap data is missing required columns: {missing}')

    raw = df.copy().reset_index(drop=True)
    raw['LapNumber'] = pd.to_numeric(raw['LapNumber'], errors='coerce')
    raw['Stint'] = pd.to_numeric(raw['Stint'], errors='coerce')
    first_stint = raw.groupby(RACE_COLUMNS, sort=False)['Stint'].transform('min')
    clean_laps = _is_clean_lap(raw)
    first_clean_lap_indices = (
        raw.loc[clean_laps]
        .groupby(STINT_COLUMNS, sort=False)['LapNumber']
        .idxmin()
    )
    first_clean_laps = raw.loc[first_clean_lap_indices]
    is_stint_out_lap = pd.Series(False, index=raw.index)
    if 'RaceId' not in raw.columns:
        later_stint_outlap_indices = first_clean_laps.index[
            first_clean_laps['Stint'].gt(first_stint.loc[first_clean_laps.index])
        ]
        is_stint_out_lap.loc[later_stint_outlap_indices] = True

    if 'RaceId' in raw.columns:
        prior_lap_median = raw.groupby(RACE_COLUMNS, sort=False)['LapTime'].transform('median')
    else:
        ordered = raw.sort_values(RACE_COLUMNS + ['LapNumber'], kind='stable')
        historical_lap_times = ordered['LapTime'].where(clean_laps.loc[ordered.index])
        prior_lap_median = (
            ordered.assign(_historical_lap_time=historical_lap_times)
            .groupby(RACE_COLUMNS, sort=False)['_historical_lap_time']
            .transform(lambda times: times.shift().expanding(min_periods=1).median())
            .reindex(raw.index)
        )
    plausible_lap_time = (
        prior_lap_median.isna()
        | raw['LapTime'].le(prior_lap_median * MAX_LAP_TIME_RATIO)
    )
    compound_max_laps = {
        'SOFT': 25,
        'MEDIUM': 35,
        'HARD': 50,
        'INTERMEDIATE': 30,
        'WET': 30,
        'UNKNOWN': 40,
    }
    compound = raw['Compound'].astype('string').str.strip().str.upper()
    tyre_life = pd.to_numeric(raw['TyreLife'], errors='coerce')
    valid_tyre_metadata = (
        compound.isin(compound_max_laps)
        & tyre_life.notna()
        & np.isfinite(tyre_life)
        & tyre_life.ge(0)
    ).fillna(False)
    pit_eligible = clean_laps & ~is_stint_out_lap & plausible_lap_time
    eligible = pit_eligible & valid_tyre_metadata
    actual_pits = _actual_pits_by_stint(raw, pit_eligible)
    observations = raw.loc[eligible].copy()
    observations = observations.sort_values(STINT_COLUMNS + ['LapNumber'])

    observations = observations.merge(actual_pits, on=STINT_COLUMNS, how='left')

    best_times_so_far = observations.groupby(STINT_COLUMNS, sort=False)['LapTime'].transform(
        lambda times: times.cummin()
    )
    observations['lap_time_delta'] = observations['LapTime'] - best_times_so_far

    observations['rolling_degradation'] = (
        observations.groupby(STINT_COLUMNS, sort=False)['lap_time_delta']
        .transform(lambda values: values.rolling(window=3, min_periods=1).mean())
    )

    max_life = compound.loc[observations.index].map(compound_max_laps).fillna(35)
    observations['tyre_age_norm'] = (
        tyre_life.loc[observations.index] / max_life
    ).clip(0, 1)

    observations[FEATURE_COLUMNS] = observations[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    observations['is_pit_lap'] = 0
    observations['FeatureVersion'] = FEATURE_VERSION
    return observations.reset_index(drop=True)


def get_stint_records(df: pd.DataFrame, min_laps: int = 3) -> list[dict]:
    """Return observations and original LapNumbers grouped by race, driver, and stint."""
    records = []
    for key, stint in df.groupby(STINT_COLUMNS, sort=False, dropna=False):
        stint = stint.sort_values('LapNumber')
        if len(stint) < min_laps:
            continue
        compounds = stint['Compound'].dropna().astype(str).str.strip().str.upper()
        compounds = compounds[compounds.ne('')]
        compound = compounds.mode().iloc[0] if not compounds.empty else 'UNKNOWN'
        record = dict(zip(STINT_COLUMNS, key if isinstance(key, tuple) else (key,)))
        record.update({
            'observations': stint[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
            'lap_numbers': stint['LapNumber'].to_numpy(dtype=np.int64),
            'compound': compound,
            'actual_pit_lap': (
                float(stint['actual_pit_lap'].dropna().iloc[0])
                if stint['actual_pit_lap'].notna().any()
                else None
            ),
        })
        records.append(record)
    return records


def scale_features(sequences: list[np.ndarray]) -> tuple[list[np.ndarray], StandardScaler]:
    """Fit a scaler on the supplied training sequences and transform each sequence."""
    if not sequences:
        raise ValueError('Cannot fit a feature scaler without training sequences.')
    scaler = StandardScaler().fit(np.vstack(sequences))
    return [scaler.transform(sequence) for sequence in sequences], scaler
