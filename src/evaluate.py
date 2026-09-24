"""
evaluate.py
-----------
Evaluation metrics for pit stop prediction.

Since we're predicting *which laps* are pit laps (not a single binary label),
we define:
    - True Positive  : model predicts pit within ±2 laps of actual pit
    - False Positive : model predicts pit with no actual pit nearby
    - False Negative : actual pit has no model prediction nearby

Also computes:
    - State sequence visualizations
    - Transition matrix heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_pit_predictions(
    predicted_laps: list,
    actual_laps: list,
    tolerance: int = 2
) -> dict:
    """
    Evaluate pit stop prediction with a tolerance window.

    A prediction is a True Positive if it falls within ±tolerance laps
    of any actual pit stop.

    Args:
        predicted_laps : list of predicted pit lap indices
        actual_laps    : list of actual pit lap indices
        tolerance      : allowed lap difference for a match

    Returns:
        dict with precision, recall, f1
    """
    tp = 0
    matched_actual = set()

    for pred in predicted_laps:
        for i, actual in enumerate(actual_laps):
            if abs(pred - actual) <= tolerance and i not in matched_actual:
                tp += 1
                matched_actual.add(i)
                break

    fp = len(predicted_laps) - tp
    fn = len(actual_laps) - len(matched_actual)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'TP': tp, 'FP': fp, 'FN': fn,
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3)
    }


def plot_tyre_state_sequence(
    lap_numbers: list,
    states: list,
    lap_times: list = None,
    actual_pit_laps: list = None,
    predicted_pit_laps: list = None,
    driver: str = '',
    save_path: str = None
):
    """
    Plot the decoded tyre state sequence alongside lap times.

    Color coding:
        FRESH      → green
        DEGRADING  → orange
        CRITICAL   → red
    """
    STATE_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    STATE_LABELS = {0: 'FRESH', 1: 'DEGRADING', 2: 'CRITICAL'}

    state_ids = [s if isinstance(s, int) else list(STATE_LABELS.values()).index(s) for s in states]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f'Tyre State Sequence — {driver}', fontsize=14, fontweight='bold')

    # Top: state sequence as colored bar
    ax1 = axes[0]
    for t, (lap, state) in enumerate(zip(lap_numbers, state_ids)):
        ax1.bar(lap, 1, color=STATE_COLORS[state], width=0.8, edgecolor='none')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=STATE_COLORS[s], label=STATE_LABELS[s]) for s in range(3)]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.set_ylabel('Tyre State')
    ax1.set_yticks([])

    # Pit markers on state plot
    if actual_pit_laps:
        for lap in actual_pit_laps:
            ax1.axvline(lap, color='blue', linestyle='--', linewidth=1.5, label='Actual Pit')
    if predicted_pit_laps:
        for lap in predicted_pit_laps:
            ax1.axvline(lap, color='purple', linestyle=':', linewidth=1.5, label='Predicted Pit')

    # Bottom: lap times
    ax2 = axes[1]
    if lap_times is not None:
        ax2.plot(lap_numbers, lap_times, color='#3498db', linewidth=1.5, marker='o', markersize=3)
        ax2.set_ylabel('Lap Time (s)')
        if actual_pit_laps:
            for lap in actual_pit_laps:
                ax2.axvline(lap, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        if predicted_pit_laps:
            for lap in predicted_pit_laps:
                ax2.axvline(lap, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Lap Number')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_transition_matrix(A: np.ndarray, save_path: str = None):
    """
    Heatmap of the learned transition matrix A.
    """
    state_labels = ['FRESH', 'DEGRADING', 'CRITICAL']
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        A, annot=True, fmt='.3f', cmap='Blues',
        xticklabels=state_labels, yticklabels=state_labels,
        ax=ax, vmin=0, vmax=1
    )
    ax.set_title('Learned HMM Transition Matrix A', fontsize=13)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def summarize_results(results: list) -> pd.DataFrame:
    """
    Summarize evaluation results across multiple drivers/races.

    Args:
        results: list of dicts from evaluate_pit_predictions()
    Returns:
        Summary DataFrame with mean metrics
    """
    df = pd.DataFrame(results)
    print("\n=== Evaluation Summary ===")
    print(df.to_string(index=False))
    print(f"\nMean Precision : {df['Precision'].mean():.3f}")
    print(f"Mean Recall    : {df['Recall'].mean():.3f}")
    print(f"Mean F1        : {df['F1'].mean():.3f}")
    return df