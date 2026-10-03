"""
features.py
-----------
Feature engineering from raw lap data.
Produces the observation matrix fed into the HMM.

Observation vector per lap:
    [lap_time_delta, tyre_age_norm, sector3_delta, rolling_degradation]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_lap_time_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each driver, compute how much slower each lap is
    compared to their personal best lap time in that stint.

    Δt = lap_time - min(lap_time for that driver & stint)
    """
    df = df.copy()

    # Assign stint number: increments each time a driver pits
    df['Stint'] = df.groupby('Driver')['is_pit_lap'].cumsum().shift(1).fillna(0).astype(int)

    # Best time per driver per stint
    best_times = df.groupby(['Driver', 'Stint'])['LapTime'].transform('min')
    df['lap_time_delta'] = df['LapTime'] - best_times

    return df


def compute_sector3_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sector 3 time delta vs driver's best sector 3 in that stint.
    Sector 3 is most sensitive to rear tyre wear.
    """
    df = df.copy()
    if 'Sector3Time' not in df.columns:
        df['sector3_delta'] = 0.0
        return df

    best_s3 = df.groupby(['Driver', 'Stint'])['Sector3Time'].transform('min')
    df['sector3_delta'] = df['Sector3Time'] - best_s3
    return df


def compute_rolling_degradation(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Rolling mean of lap_time_delta over last `window` laps per driver.
    Captures the trend of tyre degradation, not just instantaneous.
    """
    df = df.copy()

    def rolling_per_driver(group):
        group = group.sort_values('LapNumber')
        group['rolling_degradation'] = (
            group['lap_time_delta']
            .rolling(window=window, min_periods=1)
            .mean()
        )
        return group

    df = df.groupby('Driver', group_keys=False).apply(rolling_per_driver, include_groups=False)
    return df


def normalize_tyre_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize TyreLife to [0, 1] range within each compound type.
    Different compounds have different expected lifespans.
    """
    df = df.copy()
    if 'TyreLife' not in df.columns:
        df['tyre_age_norm'] = 0.0
        return df

    # Approximate max laps per compound
    compound_max = {'SOFT': 25, 'MEDIUM': 35, 'HARD': 50, 'INTERMEDIATE': 30, 'WET': 30}
    default_max = 35

    df['tyre_age_norm'] = df.apply(
        lambda row: row['TyreLife'] / compound_max.get(str(row.get('Compound', '')), default_max),
        axis=1
    ).clip(0, 1)

    return df


def build_observation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Returns DataFrame with columns:
        Driver, LapNumber, Stint, is_pit_lap,
        lap_time_delta, sector3_delta, rolling_degradation, tyre_age_norm
    """
    df = df.copy()

    # Drop laps with missing lap times (in/out laps, safety car weirdness)
    df = df.dropna(subset=['LapTime'])
    df = df[df['LapTime'] > 0]

    # Remove obvious outliers (lap time > 2x median per driver suggests red flag etc.)
    median_lap = df.groupby('Driver')['LapTime'].transform('median')
    df = df[df['LapTime'] < 2 * median_lap]

    # Feature computation
    df = compute_lap_time_delta(df)
    df = compute_sector3_delta(df)
    df = compute_rolling_degradation(df)
    df = normalize_tyre_age(df)

    # Fill any remaining NaNs
    feature_cols = ['lap_time_delta', 'sector3_delta', 'rolling_degradation', 'tyre_age_norm']
    df[feature_cols] = df[feature_cols].fillna(0)

    return df


def get_driver_sequences(df: pd.DataFrame, driver: str) -> tuple:
    """
    Extract the observation sequence for one driver,
    split by stint (each stint is an independent sequence for the HMM).

    Returns:
        sequences : list of np.arrays, shape (n_laps, n_features) per stint
        pit_laps  : list of pit lap numbers for this driver
        compounds : list of compound strings per stint e.g. ['SOFT', 'MEDIUM']
    """
    feature_cols = ['lap_time_delta', 'sector3_delta', 'rolling_degradation', 'tyre_age_norm']

    driver_df = df[df['Driver'] == driver].sort_values('LapNumber')
    pit_laps = driver_df[driver_df['is_pit_lap'] == 1]['LapNumber'].tolist()

    sequences = []
    compounds = []
    for stint_id, stint_df in driver_df.groupby('Stint'):
        stint_df = stint_df.sort_values('LapNumber')
        obs = stint_df[feature_cols].values.astype(np.float64)
        if len(obs) >= 3:  # Skip very short stints
            sequences.append(obs)
            # Most common compound in this stint
            if 'Compound' in stint_df.columns:
                compound = stint_df['Compound'].mode()[0]
            else:
                compound = 'UNKNOWN'
            compounds.append(str(compound).upper())

    return sequences, pit_laps, compounds


def get_sequences_by_compound(df: pd.DataFrame) -> dict:
    """
    Group all driver stints by tyre compound.
    Returns dict: { 'SOFT': [seq1, seq2, ...], 'MEDIUM': [...], 'HARD': [...] }

    This allows training a separate HMM per compound, since degradation
    rates differ significantly across Soft/Medium/Hard.
    """
    compound_seqs = {}

    for driver in df['Driver'].unique():
        seqs, _, compounds = get_driver_sequences(df, driver)
        for seq, compound in zip(seqs, compounds):
            compound = compound if compound in ('SOFT', 'MEDIUM', 'HARD') else 'MEDIUM'
            if compound not in compound_seqs:
                compound_seqs[compound] = []
            compound_seqs[compound].append(seq)

    for compound, seqs in compound_seqs.items():
        print(f"  {compound}: {len(seqs)} stints")

    return compound_seqs


def scale_features(sequences: list) -> tuple:
    """
    StandardScale features across all sequences.
    Returns scaled sequences and the fitted scaler (save for inference).
    """
    all_obs = np.vstack(sequences)
    scaler = StandardScaler()
    scaler.fit(all_obs)

    scaled = [scaler.transform(seq) for seq in sequences]
    return scaled, scaler


if __name__ == '__main__':
    # Quick test with dummy data
    dummy = pd.DataFrame({
        'Driver': ['VER'] * 20 + ['HAM'] * 20,
        'LapNumber': list(range(1, 21)) * 2,
        'LapTime': [90 + i * 0.1 + np.random.randn() * 0.2 for i in range(20)] * 2,
        'Sector3Time': [28 + i * 0.05 for i in range(20)] * 2,
        'TyreLife': list(range(1, 21)) * 2,
        'Compound': ['SOFT'] * 40,
        'is_pit_lap': [0] * 15 + [1] + [0] * 4 + [0] * 15 + [1] + [0] * 4,
    })

    result = build_observation_matrix(dummy)
    print(result[['Driver', 'LapNumber', 'lap_time_delta', 'tyre_age_norm', 'rolling_degradation']].head(10))