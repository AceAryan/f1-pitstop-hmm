"""
evaluate.py
-----------
Step 3: Load trained models and evaluate pit stop prediction.

Improvements over v1:
    - Uses compound-specific HMM per driver stint
    - State smoothing: require N consecutive CRITICAL laps before predicting pit
    - Evaluation with tolerance window (±2 laps)
    - Saves per-driver state sequence plots + transition matrix heatmaps

Usage:
    python evaluate.py
    python evaluate.py --smooth 3 --tolerance 3
    python evaluate.py --drivers VER HAM LEC
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import get_driver_sequences
from src.hmm import STATE_NAMES

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='F1 HMM - Evaluation')
parser.add_argument('--data', type=str, default='data/processed.parquet')
parser.add_argument('--models_dir', type=str, default='results/models')
parser.add_argument('--smooth', type=int, default=3,
                    help='Consecutive CRITICAL laps needed before predicting pit (default 3)')
parser.add_argument('--tolerance', type=int, default=2,
                    help='Allowed lap difference for TP match (default +/-2)')
parser.add_argument('--drivers', nargs='+', default=None,
                    help='Specific drivers to evaluate (default: top 10 by lap count)')
parser.add_argument('--plot_drivers', nargs='+', default=None,
                    help='Drivers to generate state sequence plots for (default: top 3)')
args = parser.parse_args()

os.makedirs('results/plots', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
print("=" * 60)
print("EVALUATE - Loading data and models")
print("=" * 60)

if not os.path.exists(args.data):
    raise FileNotFoundError(f"{args.data} not found. Run preprocess.py first.")

feat_df = pd.read_parquet(args.data)

# ── Load models ───────────────────────────────────────────────────────
models = {}
scalers = {}
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    path = os.path.join(args.models_dir, f'hmm_{compound.lower()}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        models[compound] = data['model']
        scalers[compound] = data['scaler']
        print(f"  Loaded model: {compound}")
    else:
        print(f"  Missing model for {compound} (skipping those stints)")

if not models:
    raise RuntimeError("No models found. Run train.py first.")

# ── Helper: smooth state sequence ────────────────────────────────────
def smooth_states(states: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Require `window` consecutive CRITICAL laps before flagging as CRITICAL.
    This reduces false positives from momentary noise spikes.

    Example with window=3:
        [0,0,2,0,2,2,2,2] → [0,0,0,0,0,0,2,2]
                                          ↑ only here we have 3 consecutive
    """
    smoothed = states.copy()
    critical = (states == 2).astype(int)
    consecutive = 0
    for t in range(len(states)):
        if critical[t]:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive < window:
            smoothed[t] = min(states[t], 1)  # demote CRITICAL → DEGRADING
    return smoothed


# ── Helper: predict pit laps from smoothed states ────────────────────
def predict_pits_from_states(states: np.ndarray) -> list:
    """
    Predict pit lap = first lap where state transitions INTO CRITICAL
    after smoothing. One prediction per CRITICAL entry per stint.
    """
    pit_laps = []
    for t in range(1, len(states)):
        if states[t] == 2 and states[t-1] != 2:
            pit_laps.append(t)
    return pit_laps


# ── Helper: evaluate with tolerance ──────────────────────────────────
def evaluate(predicted: list, actual: list, tolerance: int) -> dict:
    tp = 0
    matched = set()
    for pred in predicted:
        for i, act in enumerate(actual):
            if abs(pred - act) <= tolerance and i not in matched:
                tp += 1
                matched.add(i)
                break
    fp = len(predicted) - tp
    fn = len(actual) - len(matched)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3)}


# ── Select drivers ────────────────────────────────────────────────────
if args.drivers:
    drivers = args.drivers
else:
    drivers = feat_df['Driver'].value_counts().head(10).index.tolist()

plot_drivers = args.plot_drivers or drivers[:3]

print(f"\nEvaluating {len(drivers)} drivers: {drivers}")
print(f"State smoothing window: {args.smooth} laps")
print(f"Pit prediction tolerance: +/-{args.tolerance} laps\n")

# ── Evaluate each driver ──────────────────────────────────────────────
all_results = []

for driver in drivers:
    seqs, actual_pit_laps, compounds = get_driver_sequences(feat_df, driver)

    if not seqs:
        continue

    all_predicted = []
    all_states_for_plot = []
    lap_offset = 0

    for seq, compound in zip(seqs, compounds):
        # Use compound-specific model, fall back to MEDIUM
        compound_key = compound if compound in models else 'MEDIUM'
        if compound_key not in models:
            lap_offset += len(seq)
            continue

        model = models[compound_key]
        scaler = scalers[compound_key]

        scaled_seq = scaler.transform(seq)
        raw_states = model.viterbi(scaled_seq)

        # Smooth: require args.smooth consecutive CRITICAL laps
        smoothed = smooth_states(raw_states, window=args.smooth)

        # Predict pits (lap index within stint → global lap offset)
        stint_pits = predict_pits_from_states(smoothed)
        all_predicted.extend([p + lap_offset for p in stint_pits])
        all_states_for_plot.extend(smoothed.tolist())

        lap_offset += len(seq)

    metrics = evaluate(all_predicted, actual_pit_laps, args.tolerance)
    metrics['Driver'] = driver
    all_results.append(metrics)

    print(f"  {driver:4s} | Pred={all_predicted} | Actual={actual_pit_laps} | "
          f"P={metrics['Precision']:.3f} R={metrics['Recall']:.3f} F1={metrics['F1']:.3f}")

    # State sequence plot for selected drivers
    if driver in plot_drivers and all_states_for_plot:
        driver_laps = feat_df[feat_df['Driver'] == driver].sort_values('LapNumber')
        lap_nums = list(range(1, len(all_states_for_plot) + 1))
        lap_times = driver_laps['LapTime'].values[:len(all_states_for_plot)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.suptitle(f'Tyre State Sequence — {driver} (smooth={args.smooth})',
                     fontsize=13, fontweight='bold')

        STATE_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
        for t, (lap, state) in enumerate(zip(lap_nums, all_states_for_plot)):
            ax1.bar(lap, 1, color=STATE_COLORS[state], width=0.8, edgecolor='none')

        from matplotlib.patches import Patch
        ax1.legend(handles=[
            Patch(facecolor='#2ecc71', label='FRESH'),
            Patch(facecolor='#f39c12', label='DEGRADING'),
            Patch(facecolor='#e74c3c', label='CRITICAL'),
        ], loc='upper right', fontsize=9)
        ax1.set_ylabel('Tyre State')
        ax1.set_yticks([])

        for lap in actual_pit_laps:
            ax1.axvline(lap, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Actual Pit')
        for lap in all_predicted:
            ax1.axvline(lap, color='purple', linestyle=':', linewidth=1.5, alpha=0.8, label='Predicted Pit')

        ax2.plot(lap_nums, lap_times, color='#3498db', linewidth=1.5, marker='o', markersize=2)
        for lap in actual_pit_laps:
            ax2.axvline(lap, color='blue', linestyle='--', linewidth=1.2, alpha=0.6)
        for lap in all_predicted:
            ax2.axvline(lap, color='purple', linestyle=':', linewidth=1.2, alpha=0.6)
        ax2.set_ylabel('Lap Time (s)')
        ax2.set_xlabel('Lap Number')

        plt.tight_layout()
        plot_path = f'results/plots/states_{driver}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot -> {plot_path}")


# ── Transition matrix heatmaps ────────────────────────────────────────
for compound, model in models.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    state_labels = ['FRESH', 'DEGRADING', 'CRITICAL']
    sns.heatmap(model.A, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=state_labels, yticklabels=state_labels,
                ax=ax, vmin=0, vmax=1)
    ax.set_title(f'Transition Matrix — {compound}', fontsize=12)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    plt.tight_layout()
    plt.savefig(f'results/plots/transition_{compound.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("EVALUATION SUMMARY")
print('='*60)

results_df = pd.DataFrame(all_results)
print(results_df[['Driver', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']].to_string(index=False))

print(f"\nMean Precision : {results_df['Precision'].mean():.3f}")
print(f"Mean Recall    : {results_df['Recall'].mean():.3f}")
print(f"Mean F1        : {results_df['F1'].mean():.3f}")

results_df.to_csv('results/evaluation_summary.csv', index=False)
print(f"\nSaved -> results/evaluation_summary.csv")
print(f"Plots -> results/plots/")