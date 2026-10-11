"""
data_loader.py
--------------
Fetches lap-by-lap telemetry from FastF1 for a given race.
Caches locally so repeated runs are fast.
"""

import fastf1
import pandas as pd
import os

# Point FastF1 to local cache — create directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race(year: int, round_name: str, session_type: str = 'R') -> pd.DataFrame:
    """
    Load lap data for a specific race session.

    Args:
        year       : Season year e.g. 2023
        round_name : Grand Prix name e.g. 'Bahrain'
        session_type: 'R' = Race, 'Q' = Qualifying

    Returns:
        DataFrame with one row per lap per driver
    """
    print(f"Loading {year} {round_name} GP ({session_type})...")
    session = fastf1.get_session(year, round_name, session_type)
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps.copy()

    # Keep only relevant columns
    cols = [
        'Driver', 'LapNumber', 'Stint', 'LapTime', 'Sector1Time', 'Sector2Time',
        'Sector3Time', 'Compound', 'TyreLife', 'FreshTyre',
        'PitInTime', 'PitOutTime', 'TrackStatus', 'IsAccurate',
        'Position', 'IsPersonalBest'
    ]
    cols = [c for c in cols if c in laps.columns]
    laps = laps[cols].copy()

    # Convert timedelta to seconds (float)
    time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in time_cols:
        if col in laps.columns:
            laps[col] = laps[col].dt.total_seconds()

    laps['Year'] = year
    laps['GrandPrix'] = round_name
    laps['is_pit_lap'] = laps['PitInTime'].notna().astype(int)

    print(f"  Loaded {len(laps)} laps across {laps['Driver'].nunique()} drivers.")
    return laps


def load_multiple_races(race_list: list) -> pd.DataFrame:
    """
    Load multiple races and concatenate.

    Args:
        race_list: list of (year, round_name) tuples
                   e.g. [(2023, 'Bahrain'), (2023, 'Monaco')]

    Returns:
        Combined DataFrame
    """
    frames = []
    for year, name in race_list:
        try:
            df = load_race(year, name)
            frames.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {year} {name}: {e}")
    
    if not frames:
        raise ValueError("No races loaded successfully.")
    
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal laps loaded: {len(combined)}")
    return combined


if __name__ == '__main__':
    # Quick test
    df = load_race(2023, 'Bahrain')
    print(df.head())
    print(df.dtypes)