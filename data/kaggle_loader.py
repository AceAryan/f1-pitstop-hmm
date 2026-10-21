"""Load the Ergast-derived Kaggle tables into the pipeline's lap schema."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_observation_matrix


_KEYS = ['raceId', 'driverId']
_KNOWN_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}


def _find_column(frame: pd.DataFrame, *names: str) -> str | None:
    columns = {str(column).lower().replace('_', ''): column for column in frame.columns}
    for name in names:
        key = name.lower().replace('_', '')
        if key in columns:
            return columns[key]
    return None


def _require_columns(frame: pd.DataFrame, table: str, **aliases: tuple[str, ...]) -> dict:
    found = {name: _find_column(frame, *choices) for name, choices in aliases.items()}
    missing = [name for name, column in found.items() if column is None]
    if missing:
        raise ValueError(f'{table} is missing required columns: {missing}.')
    return found


def _read_compound_data(data_path: Path, results: pd.DataFrame) -> pd.DataFrame:
    """Return available per-lap/per-stint tyre metadata, or an empty table."""
    tyre_path = data_path / 'tyres.csv'
    source = pd.read_csv(tyre_path) if tyre_path.exists() else results
    keys = {
        'raceId': _find_column(source, 'raceId'),
        'driverId': _find_column(source, 'driverId'),
    }
    compound_column = _find_column(source, 'compound', 'tyreCompound')
    stint_column = _find_column(source, 'stint')
    lap_column = _find_column(source, 'lap', 'lapNumber')
    life_column = _find_column(source, 'TyreLife', 'tyreAge', 'life')
    if not all(keys.values()) or compound_column is None:
        return pd.DataFrame(columns=_KEYS + ['Stint', 'LapNumber', 'Compound', 'TyreLife'])

    columns = [keys['raceId'], keys['driverId'], compound_column]
    rename = {
        keys['raceId']: 'raceId',
        keys['driverId']: 'driverId',
        compound_column: 'Compound',
    }
    if stint_column is not None:
        columns.append(stint_column)
        rename[stint_column] = 'Stint'
    if lap_column is not None:
        columns.append(lap_column)
        rename[lap_column] = 'LapNumber'
    if life_column is not None:
        columns.append(life_column)
        rename[life_column] = 'TyreLife'
    metadata = source[columns].rename(columns=rename).copy()
    metadata['Compound'] = (
        metadata['Compound'].astype('string').str.strip().str.upper()
    )
    metadata = metadata[metadata['Compound'].notna()]
    return metadata.drop_duplicates(
        [column for column in ('raceId', 'driverId', 'Stint', 'LapNumber') if column in metadata],
        keep='last',
    )


def _assemble_raw_race(
    race: pd.Series,
    lap_times: pd.DataFrame,
    pit_stops: pd.DataFrame,
    compound_data: pd.DataFrame,
) -> pd.DataFrame:
    race_id = race['raceId']
    laps = lap_times.loc[lap_times['raceId'].eq(race_id)].copy()
    if laps.empty:
        return pd.DataFrame()

    laps['LapNumber'] = pd.to_numeric(laps['lap'], errors='coerce')
    laps['LapTime'] = pd.to_numeric(laps['milliseconds'], errors='coerce') / 1000.0
    stops = pit_stops.loc[pit_stops['raceId'].eq(race_id), _KEYS + ['lap']].copy()
    stops['lap'] = pd.to_numeric(stops['lap'], errors='coerce')
    stops = stops.dropna(subset=['lap']).drop_duplicates(_KEYS + ['lap'])

    rows = []
    for (driver_id, _), driver_laps in laps.groupby(['driverId', 'raceId'], sort=False):
        driver_laps = driver_laps.sort_values('LapNumber', kind='stable').copy()
        driver_stops = np.sort(
            stops.loc[stops['driverId'].eq(driver_id), 'lap'].to_numpy(dtype=float)
        )
        lap_numbers = driver_laps['LapNumber'].to_numpy(dtype=float)
        driver_laps['Stint'] = np.searchsorted(driver_stops, lap_numbers, side='left') + 1
        driver_laps['IsPitLap'] = driver_laps['LapNumber'].isin(driver_stops)
        driver_laps['Year'] = int(race['year'])
        driver_laps['GrandPrix'] = str(race['name'])
        driver_laps['Driver'] = str(driver_id)
        driver_laps['RaceId'] = race_id
        driver_laps['DriverId'] = driver_id
        rows.append(driver_laps)

    raw = pd.concat(rows, ignore_index=True)
    raw_lap_count = len(raw)

    metadata = compound_data.loc[compound_data['raceId'].eq(race_id)].copy()
    if not metadata.empty:
        merge_keys = ['raceId', 'driverId']
        if 'Stint' in metadata.columns:
            merge_keys.append('Stint')
        if 'LapNumber' in metadata.columns:
            merge_keys.append('LapNumber')
        metadata = metadata.drop_duplicates(merge_keys, keep='last')
        raw = raw.merge(metadata, on=merge_keys, how='left', suffixes=('', '_tyre'))
        if 'Compound_tyre' in raw:
            raw['Compound'] = raw['Compound_tyre'].combine_first(raw.get('Compound'))
            raw = raw.drop(columns='Compound_tyre')
        if 'TyreLife_tyre' in raw:
            raw['TyreLife'] = raw['TyreLife_tyre'].combine_first(raw.get('TyreLife'))
            raw = raw.drop(columns='TyreLife_tyre')

    if 'Compound' not in raw:
        raw['Compound'] = 'UNKNOWN'
    else:
        compound = raw['Compound'].astype('string').str.strip().str.upper()
        raw['Compound'] = compound.where(compound.isin(_KNOWN_COMPOUNDS), 'UNKNOWN')
    if 'TyreLife' not in raw:
        raw['TyreLife'] = np.nan
    raw['TyreLife'] = pd.to_numeric(raw['TyreLife'], errors='coerce')
    start_of_stint = raw.groupby(['raceId', 'driverId', 'Stint'])['LapNumber'].transform('min')
    inferred_life = pd.Series(
        np.where(
            raw['Stint'].eq(1),
            raw['LapNumber'],
            raw['LapNumber'] - start_of_stint + 1,
        ),
        index=raw.index,
    )
    raw['TyreLife'] = raw['TyreLife'].fillna(inferred_life)

    pit_laps = raw.loc[raw['IsPitLap'], ['raceId', 'driverId', 'LapNumber']]
    raw['_pit_event'] = raw['IsPitLap']
    raw['_pit_out'] = raw.set_index(['raceId', 'driverId', 'LapNumber']).index.isin(
        pd.MultiIndex.from_frame(
            pit_laps.assign(LapNumber=pit_laps['LapNumber'] + 1)
        )
    )
    raw['PitInTime'] = pd.NaT
    raw.loc[raw['_pit_event'], 'PitInTime'] = pd.Timestamp('2000-01-01')
    raw['PitOutTime'] = pd.NaT
    raw.loc[raw['_pit_out'], 'PitOutTime'] = pd.Timestamp('2000-01-01')
    raw['TrackStatus'] = '1'
    raw['IsAccurate'] = True

    compounds = set(raw['Compound'].dropna().str.upper())
    if {'INTERMEDIATE', 'WET'} & compounds:
        return pd.DataFrame()

    raw = raw.drop(columns=['_pit_event', '_pit_out'], errors='ignore')
    raw.attrs['raw_lap_count'] = raw_lap_count
    return raw


def load_season(
    year: int,
    data_dir: str | Path,
    *,
    include_raw: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load clean observations, optionally returning the complete dry-race laps too."""
    directory = Path(data_dir)
    required_files = ('races.csv', 'lap_times.csv', 'pit_stops.csv', 'results.csv')
    missing_files = [name for name in required_files if not (directory / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f'Missing Kaggle data files in {directory}: {", ".join(missing_files)}.'
        )

    races_source = pd.read_csv(directory / 'races.csv')
    races_columns = _require_columns(
        races_source,
        'races.csv',
        raceId=('raceId',),
        year=('year',),
        name=('name', 'raceName'),
    )
    races = races_source.rename(columns={column: key for key, column in races_columns.items()})
    races = races[pd.to_numeric(races['year'], errors='coerce').eq(year)].copy()
    if races.empty:
        raise ValueError(f'No races for season {year} were found in {directory / "races.csv"}.')

    lap_source = pd.read_csv(directory / 'lap_times.csv')
    lap_columns = _require_columns(
        lap_source,
        'lap_times.csv',
        raceId=('raceId',),
        driverId=('driverId',),
        lap=('lap', 'LapNumber'),
        milliseconds=('milliseconds',),
    )
    lap_times = lap_source.rename(columns={column: key for key, column in lap_columns.items()})
    lap_times = lap_times[lap_times['raceId'].isin(races['raceId'])].copy()

    pit_source = pd.read_csv(directory / 'pit_stops.csv')
    pit_columns = _require_columns(
        pit_source,
        'pit_stops.csv',
        raceId=('raceId',),
        driverId=('driverId',),
        lap=('lap', 'LapNumber'),
    )
    pit_stops = pit_source.rename(columns={column: key for key, column in pit_columns.items()})
    pit_stops = pit_stops[pit_stops['raceId'].isin(races['raceId'])].copy()

    results = pd.read_csv(directory / 'results.csv')
    result_columns = _require_columns(
        results, 'results.csv', raceId=('raceId',), driverId=('driverId',)
    )
    results = results.rename(columns={column: key for key, column in result_columns.items()})
    results = results[results['raceId'].isin(races['raceId'])]
    lap_times = lap_times.merge(results[_KEYS].drop_duplicates(), on=_KEYS, how='inner')
    compound_data = _read_compound_data(directory, results)

    frames = []
    raw_frames = []
    raw_lap_count = 0
    race_count = 0
    for _, race in races.sort_values(['year', 'raceId']).iterrows():
        raw = _assemble_raw_race(race, lap_times, pit_stops, compound_data)
        if raw.empty:
            continue
        raw_lap_count += int(raw.attrs.get('raw_lap_count', len(raw)))
        race_count += 1
        raw_frames.append(raw)
        observations = build_observation_matrix(raw)
        if observations.empty:
            continue
        observations['RaceId'] = race['raceId']
        observations['LapTime'] = pd.to_numeric(observations['LapTime'], errors='raise')
        frames.append(observations)

    if not frames:
        raise ValueError(f'No clean laps were available for season {year}.')
    combined = pd.concat(frames, ignore_index=True)
    combined.attrs['raw_lap_count'] = raw_lap_count
    combined.attrs['race_count'] = race_count
    if include_raw:
        raw_combined = pd.concat(raw_frames, ignore_index=True)
        return combined, raw_combined
    return combined
