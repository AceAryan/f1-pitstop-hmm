"""
preprocess.py
-------------
Step 1: Load raw race data and engineer features.
Saves processed data to data/processed.parquet

Usage:
    python preprocess.py
    python preprocess.py --year 2022 --gp Bahrain Australia Monaco Spa Silverstone
    python preprocess.py --year 2022  # uses default GP list
"""

import os
import argparse
import pandas as pd

from src.data_loader import load_race
from src.features import build_observation_matrix

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='F1 HMM — Preprocessing')
parser.add_argument('--year', type=int, default=None)
parser.add_argument('--gp', nargs='+', default=None)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────
DEFAULT_RACES = [
    (2022, 'Bahrain'),
    (2022, 'Australia'),
    (2022, 'Monaco'),
    (2022, 'Silverstone'),
    (2022, 'Spa'),
    (2022, 'Monza'),
    (2023, 'Bahrain'),
    (2023, 'Australia'),
    (2023, 'Monaco'),
]

DEFAULT_GPS = ['Bahrain', 'Australia', 'Monaco', 'Silverstone', 'Spa', 'Monza']

if args.year and args.gp:
    RACES = [(args.year, gp) for gp in args.gp]
elif args.year:
    RACES = [(args.year, gp) for gp in DEFAULT_GPS]
else:
    RACES = DEFAULT_RACES

os.makedirs('data', exist_ok=True)
OUT_PATH = 'data/processed.parquet'

# ── Load ──────────────────────────────────────────────────────────────
print("=" * 60)
print("PREPROCESS — Loading race data")
print(f"Races: {RACES}")
print("=" * 60)

all_laps = []
for year, gp in RACES:
    try:
        df = load_race(year, gp)
        all_laps.append(df)
    except Exception as e:
        print(f"  Skipping {year} {gp}: {e}")

if not all_laps:
    raise RuntimeError("No races loaded.")

raw_df = pd.concat(all_laps, ignore_index=True)
print(f"\nRaw laps: {len(raw_df)} across {raw_df['Driver'].nunique()} drivers")

# ── Feature engineering ───────────────────────────────────────────────
print("\nEngineering features...")
feat_df = build_observation_matrix(raw_df)

# ── Save ──────────────────────────────────────────────────────────────
feat_df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved processed data → {OUT_PATH}")
print(f"Shape: {feat_df.shape}")
print(f"Columns: {list(feat_df.columns)}")
print(f"Compounds: {feat_df['Compound'].value_counts().to_dict() if 'Compound' in feat_df.columns else 'N/A'}")