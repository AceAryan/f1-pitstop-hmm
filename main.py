"""
main.py
-------
Entry point for the F1 Pit Stop HMM project.

Usage:
    python main.py                        # uses default RACES list below
    python main.py --year 2022            # all default GPs for a specific year
    python main.py --year 2021 --gp Spa Monaco Silverstone

FastF1 supports seasons from 2018 onwards reliably.
Available years: 2018 – 2024 (2025 partial, depending on FastF1 updates)

This script:
    1. Loads F1 race data via FastF1
    2. Engineers features (lap time delta, tyre age, etc.)
    3. Trains the HMM using Baum-Welch on multiple drivers
    4. Decodes tyre states using Viterbi for each driver
    5. Evaluates pit stop prediction accuracy
    6. Saves visualizations to results/
"""

import os
import argparse
import numpy as np
import pandas as pd

from src.data_loader import load_race
from src.features import build_observation_matrix, get_driver_sequences, scale_features
from src.hmm import GaussianHMM
from src.evaluate import (
    evaluate_pit_predictions,
    plot_tyre_state_sequence,
    plot_transition_matrix,
    summarize_results
)

# ── Argument parsing ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='F1 Pit Stop HMM')
parser.add_argument('--year', type=int, default=None,
                    help='Season year (2018–2024). Overrides RACES list.')
parser.add_argument('--gp', nargs='+', default=None,
                    help='Grand Prix names e.g. --gp Bahrain Monaco Spa')
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────
# Default races — edit freely or override via CLI flags above
DEFAULT_RACES = [
    (2023, 'Bahrain'),
    (2023, 'Australia'),
    (2023, 'Monaco'),
]

# Default GPs used when --year is passed without --gp
DEFAULT_GPS = ['Bahrain', 'Australia', 'Monaco', 'Silverstone', 'Spa']

if args.year and args.gp:
    RACES = [(args.year, gp) for gp in args.gp]
elif args.year:
    RACES = [(args.year, gp) for gp in DEFAULT_GPS]
else:
    RACES = DEFAULT_RACES

print(f"Races to load: {RACES}")
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading race data")
print("=" * 60)

all_laps = []
for year, gp in RACES:
    try:
        df = load_race(year, gp)
        all_laps.append(df)
    except Exception as e:
        print(f"Skipping {year} {gp}: {e}")

if not all_laps:
    raise RuntimeError("No race data loaded. Check your internet connection or cache.")

raw_df = pd.concat(all_laps, ignore_index=True)
print(f"\nTotal laps: {len(raw_df)}")

# ── 2. Feature Engineering ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Feature engineering")
print("=" * 60)

feat_df = build_observation_matrix(raw_df)
print(feat_df[['Driver', 'LapNumber', 'lap_time_delta', 'tyre_age_norm',
               'rolling_degradation', 'sector3_delta']].head(10))

# ── 3. Prepare sequences ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Preparing observation sequences")
print("=" * 60)

# Use top drivers with enough laps
top_drivers = feat_df['Driver'].value_counts().head(10).index.tolist()
print(f"Drivers selected: {top_drivers}")

all_sequences = []
driver_data = {}

for driver in top_drivers:
    seqs, pit_laps = get_driver_sequences(feat_df, driver)
    if seqs:
        all_sequences.extend(seqs)
        driver_data[driver] = (seqs, pit_laps)

print(f"Total stints for training: {len(all_sequences)}")

# Scale features
scaled_sequences, scaler = scale_features(all_sequences)

# ── 4. Train HMM ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Training HMM (Baum-Welch)")
print("=" * 60)

model = GaussianHMM(n_states=3, n_iter=30, tol=1e-4)
model.fit(scaled_sequences)

# Save transition matrix plot
plot_transition_matrix(
    model.A,
    save_path=os.path.join(RESULTS_DIR, 'transition_matrix.png')
)

# ── 5. Decode + Evaluate ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Viterbi decoding & evaluation")
print("=" * 60)

all_results = []

for driver in top_drivers:
    if driver not in driver_data:
        continue

    seqs, actual_pit_laps = driver_data[driver]
    driver_laps = feat_df[feat_df['Driver'] == driver].sort_values('LapNumber')

    predicted_pit_laps = []
    offset = 0

    for seq in seqs:
        scaled_seq = scaler.transform(seq)
        pits = model.predict_pit_laps(scaled_seq)
        predicted_pit_laps.extend([p + offset for p in pits])
        offset += len(seq)

    metrics = evaluate_pit_predictions(predicted_pit_laps, actual_pit_laps)
    metrics['Driver'] = driver
    all_results.append(metrics)

    print(f"  {driver}: Predicted={predicted_pit_laps}, Actual={actual_pit_laps} | {metrics}")

    # State sequence plot for first driver
    if driver == top_drivers[0] and seqs:
        scaled_seq = scaler.transform(seqs[0])
        states = model.decode(scaled_seq)
        lap_nums = list(range(1, len(states) + 1))
        lap_times = driver_laps['LapTime'].values[:len(states)]

        plot_tyre_state_sequence(
            lap_nums, states, lap_times,
            actual_pit_laps=actual_pit_laps,
            predicted_pit_laps=predicted_pit_laps,
            driver=driver,
            save_path=os.path.join(RESULTS_DIR, f'state_sequence_{driver}.png')
        )

# ── 6. Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Final Results")
print("=" * 60)

summary = summarize_results(all_results)
summary.to_csv(os.path.join(RESULTS_DIR, 'evaluation_summary.csv'), index=False)

print(f"\nAll outputs saved to '{RESULTS_DIR}/'")
print("Done!")