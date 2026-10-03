"""
main.py
-------
Single entry point for the F1 Pit Stop HMM pipeline.

Usage:
    python main.py                          # run full pipeline (all 3 steps)
    python main.py --preprocess             # only load data + engineer features
    python main.py --train                  # only train HMMs
    python main.py --evaluate               # only evaluate + plot
    python main.py --preprocess --train     # chain specific steps

Preprocess options:
    --year 2022                             # season year (2018–2024)
    --gp Bahrain Monaco Spa                 # specific Grand Prix names

Train options:
    --compounds SOFT MEDIUM HARD
    --n_iter 50
    --n_states 3

Evaluate options:
    --smooth 3                              # consecutive CRITICAL laps before predicting pit
    --tolerance 2                           # ±lap window for TP match
    --drivers VER HAM LEC                   # specific drivers (default: top 10)
    --plot_drivers VER HAM                  # drivers to save state plots for
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from src.data_loader import load_race
from src.features import (
    build_observation_matrix,
    get_sequences_by_compound,
    get_driver_sequences,
    scale_features,
)
from src.hmm import GaussianHMM, STATE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────
DATA_PATH   = 'data/processed.csv'
MODELS_DIR  = 'results/models'
PLOTS_DIR   = 'results/plots'
RESULTS_CSV = 'results/evaluation_summary.csv'

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


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════

def run_preprocess(year=None, gp=None):
    """Load race data via FastF1, engineer features, save to parquet."""

    if year and gp:
        races = [(year, g) for g in gp]
    elif year:
        races = [(year, g) for g in DEFAULT_GPS]
    else:
        races = DEFAULT_RACES

    print("=" * 60)
    print("STEP 1: PREPROCESS")
    print(f"Races: {races}")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    all_laps = []
    for yr, gp_name in races:
        try:
            df = load_race(yr, gp_name)
            all_laps.append(df)
        except Exception as e:
            print(f"  Skipping {yr} {gp_name}: {e}")

    if not all_laps:
        raise RuntimeError("No races loaded. Check internet or cache.")

    raw_df = pd.concat(all_laps, ignore_index=True)
    print(f"\nRaw laps: {len(raw_df)} across {raw_df['Driver'].nunique()} drivers")

    print("Engineering features...")
    feat_df = build_observation_matrix(raw_df)

    feat_df.to_csv(DATA_PATH, index=False)
    print(f"Saved -> {DATA_PATH}  ({feat_df.shape[0]} rows, {feat_df.shape[1]} cols)")
    if 'Compound' in feat_df.columns:
        print(f"Compounds: {feat_df['Compound'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN
# ══════════════════════════════════════════════════════════════════════

def run_train(compounds=None, n_iter=50, n_states=3):
    """Train one HMM per tyre compound. Saves pickles to results/models/."""

    if compounds is None:
        compounds = ['SOFT', 'MEDIUM', 'HARD']

    print("\n" + "=" * 60)
    print("STEP 2: TRAIN")
    print(f"Compounds: {compounds} | n_states={n_states} | n_iter={n_iter}")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found - run with --preprocess first.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    feat_df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(feat_df)} laps, {feat_df['Driver'].nunique()} drivers")

    print("\nGrouping stints by compound...")
    compound_seqs = get_sequences_by_compound(feat_df)

    for compound in compounds:
        seqs = compound_seqs.get(compound, [])
        if len(seqs) < 5:
            print(f"\n[{compound}] Only {len(seqs)} stints - skipping (need >= 5)")
            continue

        print(f"\n{'─'*60}")
        print(f"Training HMM: {compound}  ({len(seqs)} stints)")
        print('─' * 60)

        scaled_seqs, scaler = scale_features(seqs)
        model = GaussianHMM(n_states=n_states, n_iter=n_iter, tol=1e-4)
        model.fit(scaled_seqs)

        path = os.path.join(MODELS_DIR, f'hmm_{compound.lower()}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        print(f"Saved -> {path}")

        labels = ['FRESH', 'DEGR', 'CRIT']
        print(f"\n[{compound}] Transition Matrix A:")
        print(f"{'':8}" + "  ".join(f"{s:>6}" for s in labels))
        for i, lbl in enumerate(labels):
            row = "  ".join(f"{model.A[i,j]:6.3f}" for j in range(n_states))
            print(f"  {lbl:6}  {row}")

    print(f"\nTraining complete -> {MODELS_DIR}/")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — EVALUATE
# ══════════════════════════════════════════════════════════════════════

def smooth_states(states: np.ndarray, window: int) -> np.ndarray:
    """
    Require `window` consecutive CRITICAL laps before flagging.
    Reduces false positives from momentary noise spikes.
    """
    smoothed = states.copy()
    consecutive = 0
    for t in range(len(states)):
        if states[t] == 2:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive < window:
            smoothed[t] = min(states[t], 1)
    return smoothed


def predict_pits_from_states(states: np.ndarray) -> list:
    """Pit lap = first lap of each CRITICAL entry after smoothing."""
    return [t for t in range(1, len(states)) if states[t] == 2 and states[t-1] != 2]


def eval_metrics(predicted: list, actual: list, tolerance: int) -> dict:
    tp, matched = 0, set()
    for pred in predicted:
        for i, act in enumerate(actual):
            if abs(pred - act) <= tolerance and i not in matched:
                tp += 1
                matched.add(i)
                break
    fp = len(predicted) - tp
    fn = len(actual) - len(matched)
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': round(p, 3), 'Recall': round(r, 3), 'F1': round(f1, 3)}


def run_evaluate(smooth=3, tolerance=2, drivers=None, plot_drivers=None):
    """Viterbi decode per driver, evaluate pit predictions, save plots."""

    print("\n" + "=" * 60)
    print("STEP 3: EVALUATE")
    print(f"Smooth={smooth} laps | Tolerance=+/-{tolerance} laps")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found - run with --preprocess first.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    feat_df = pd.read_csv(DATA_PATH)

    # Load models
    models, scalers = {}, {}
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        path = os.path.join(MODELS_DIR, f'hmm_{compound.lower()}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            models[compound]  = data['model']
            scalers[compound] = data['scaler']
            print(f"  Loaded model: {compound}")
        else:
            print(f"  Missing: {compound} (skipping those stints)")

    if not models:
        raise RuntimeError("No models found - run with --train first.")

    all_drivers = drivers or feat_df['Driver'].value_counts().head(10).index.tolist()
    plot_for    = plot_drivers or all_drivers[:3]

    print(f"\nDrivers: {all_drivers}")
    all_results = []

    for driver in all_drivers:
        seqs, actual_pits, compounds = get_driver_sequences(feat_df, driver)
        if not seqs:
            continue

        predicted, state_trace = [], []
        offset = 0

        for seq, compound in zip(seqs, compounds):
            key = compound if compound in models else 'MEDIUM'
            if key not in models:
                offset += len(seq)
                continue

            scaled = scalers[key].transform(seq)
            raw    = models[key].viterbi(scaled)
            smoothed_states = smooth_states(raw, window=smooth)
            pits   = predict_pits_from_states(smoothed_states)

            predicted.extend([p + offset for p in pits])
            state_trace.extend(smoothed_states.tolist())
            offset += len(seq)

        metrics = eval_metrics(predicted, actual_pits, tolerance)
        metrics['Driver'] = driver
        all_results.append(metrics)

        print(f"  {driver:4s} | pred={predicted} actual={actual_pits} | "
              f"P={metrics['Precision']:.3f} R={metrics['Recall']:.3f} F1={metrics['F1']:.3f}")

        # ── State sequence plot ───────────────────────────────────────
        if driver in plot_for and state_trace:
            driver_laps = feat_df[feat_df['Driver'] == driver].sort_values('LapNumber')
            lap_nums  = list(range(1, len(state_trace) + 1))
            lap_times = driver_laps['LapTime'].values[:len(state_trace)]

            STATE_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            fig.suptitle(f'Tyre State Sequence — {driver}', fontsize=13, fontweight='bold')

            for lap, state in zip(lap_nums, state_trace):
                ax1.bar(lap, 1, color=STATE_COLORS[state], width=0.8, edgecolor='none')

            ax1.legend(handles=[
                Patch(facecolor='#2ecc71', label='FRESH'),
                Patch(facecolor='#f39c12', label='DEGRADING'),
                Patch(facecolor='#e74c3c', label='CRITICAL'),
            ], loc='upper right', fontsize=9)
            ax1.set_ylabel('Tyre State')
            ax1.set_yticks([])

            for lap in actual_pits:
                ax1.axvline(lap, color='blue',   linestyle='--', lw=1.5, alpha=0.8)
            for lap in predicted:
                ax1.axvline(lap, color='purple', linestyle=':',  lw=1.5, alpha=0.8)

            ax1.plot([], [], color='blue',   linestyle='--', label='Actual pit')
            ax1.plot([], [], color='purple', linestyle=':',  label='Predicted pit')
            ax1.legend(loc='upper left', fontsize=8)

            ax2.plot(lap_nums, lap_times, color='#3498db', lw=1.5, marker='o', markersize=2)
            for lap in actual_pits:
                ax2.axvline(lap, color='blue',   linestyle='--', lw=1.2, alpha=0.6)
            for lap in predicted:
                ax2.axvline(lap, color='purple', linestyle=':',  lw=1.2, alpha=0.6)
            ax2.set_ylabel('Lap Time (s)')
            ax2.set_xlabel('Lap Number')

            plt.tight_layout()
            out = os.path.join(PLOTS_DIR, f'states_{driver}.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot -> {out}")

    # ── Transition matrix heatmaps ────────────────────────────────────
    for compound, model in models.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ['FRESH', 'DEGRADING', 'CRITICAL']
        sns.heatmap(model.A, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f'Transition Matrix — {compound}', fontsize=12)
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'transition_{compound.lower()}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print('=' * 60)
    results_df = pd.DataFrame(all_results)
    print(results_df[['Driver','TP','FP','FN','Precision','Recall','F1']].to_string(index=False))
    print(f"\nMean Precision : {results_df['Precision'].mean():.3f}")
    print(f"Mean Recall    : {results_df['Recall'].mean():.3f}")
    print(f"Mean F1        : {results_df['F1'].mean():.3f}")

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved -> {RESULTS_CSV}")
    print(f"Plots -> {PLOTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='F1 Pit Stop HMM Pipeline',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py                                # full pipeline
  python main.py --preprocess                   # only preprocess
  python main.py --train                        # only train
  python main.py --evaluate                     # only evaluate
  python main.py --preprocess --train           # preprocess then train
  python main.py --preprocess --year 2021 --gp Bahrain Monaco
  python main.py --train --n_iter 80
  python main.py --evaluate --smooth 4 --drivers VER HAM LEC
        """
    )

    # Step flags
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing step')
    parser.add_argument('--train',      action='store_true', help='Run training step')
    parser.add_argument('--evaluate',   action='store_true', help='Run evaluation step')

    # Preprocess options
    parser.add_argument('--year', type=int, default=None,
                        help='Season year for --preprocess (2018–2024)')
    parser.add_argument('--gp', nargs='+', default=None,
                        help='GP names for --preprocess e.g. --gp Bahrain Monaco Spa')

    # Train options
    parser.add_argument('--compounds', nargs='+', default=['SOFT', 'MEDIUM', 'HARD'],
                        help='Compounds to train HMMs for')
    parser.add_argument('--n_iter',   type=int, default=50,
                        help='Baum-Welch iterations (default 50)')
    parser.add_argument('--n_states', type=int, default=3,
                        help='Number of hidden states (default 3)')

    # Evaluate options
    parser.add_argument('--smooth',       type=int,    default=3,
                        help='Consecutive CRITICAL laps before predicting pit (default 3)')
    parser.add_argument('--tolerance',    type=int,    default=2,
                        help='Lap tolerance for TP match (default +/-2)')
    parser.add_argument('--drivers',      nargs='+',   default=None,
                        help='Driver codes to evaluate (default: top 10)')
    parser.add_argument('--plot_drivers', nargs='+',   default=None,
                        help='Drivers to save state plots for (default: top 3)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # If no step flag given → run all three
    run_all = not any([args.preprocess, args.train, args.evaluate])

    if run_all or args.preprocess:
        run_preprocess(year=args.year, gp=args.gp)

    if run_all or args.train:
        run_train(
            compounds=args.compounds,
            n_iter=args.n_iter,
            n_states=args.n_states,
        )

    if run_all or args.evaluate:
        run_evaluate(
            smooth=args.smooth,
            tolerance=args.tolerance,
            drivers=args.drivers,
            plot_drivers=args.plot_drivers,
        )

    print("\n Done.")