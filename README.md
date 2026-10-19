# F1 Pit-Stop HMM

Infer tyre-degradation states from Formula 1 lap data and compare pit-stop
predictions against two baselines: a tyre-age threshold and logistic regression.

## Pipeline

The single entry point is `main.py`; processed observations use CSV throughout.

```text
python main.py --preprocess --years 2020 2021 2022 2023 2024
python main.py --tune                           # fit on 2020-2022; select using 2023 only
python main.py --train --test_year 2024         # refit on 2020-2023 using selected settings
python main.py --evaluate --test_year 2024      # one-time final test
python main.py --preprocess --year 2022 --gp Bahrain Monaco
```

The final experiment is chronological: candidate models are fit on 2020–2022,
all threshold and state-count selection uses 2023, and final models are refit
on 2020–2023 before the 2024 test is evaluated once. `--tune` selects the
terminal-state posterior threshold (0.3–0.9), tyre-age threshold (0.60–0.95),
and state count (2, 3, or 4) by pooled 2023 F1 at +/-2 laps. It also tunes the
no-tyre-age HMM ablation on the same validation season. Fitted scalers and
models use training seasons only. The 2024 final evaluation is guarded against
repeat runs; it requires final models trained with the saved 2023 configuration.
Preprocessing defaults to dry-tyre races in 2020–2023; pass `--years` to add
2024 or select another scope.

FastF1 observations are grouped by **Year, Grand Prix, Driver, and FastF1 Stint**.
The pipeline retains real `LapNumber` values and pit-in lap targets, while
excluding pit-in/out laps, inaccurate laps, and non-green track-status laps from
model observations. Entire races are excluded before stint extraction if any
driver used intermediate or wet-compound tyres. It also excludes the first
valid lap after each subsequent stint boundary, since FastF1 may omit
`PitOutTime` or the pit-exit lap may have no valid lap time. Extreme lap-time
outliers (over 1.35 times that driver's expanding median of prior clean laps)
are removed without using future lap times. Laps missing a compound or valid
tyre age are excluded rather than assigned a fresh-tyre age. Lap-time and
sector deltas use only the best time observed up to that lap; they do not use
future laps from the stint. Regenerate existing processed data with
`--preprocess`; older or feature-version-mismatched CSV files are rejected with
an actionable error.

## Hidden states and prediction

| State | Meaning |
|---|---|
| 0 | FRESH |
| 1 through N-2 | DEGRADING (no intermediate state for a 2-state model) |
| N-1 | CRITICAL |

The HMM starts in state 0, uses left-to-right state transitions, and initializes
state emissions in increasing order of its final observation feature (tyre age
for the full model; rolling degradation for the ablation). Evaluation uses
forward-filtered probabilities (`P(state_t | laps through t)`), not Viterbi
decoding, so a prediction never uses future laps from the same stint.

## Baselines and output

- **Tyre-age rule:** predict once per stint at the threshold selected on 2023.
- **Logistic regression:** predict within a three-lap horizon, trained using
  training-season pit targets.
- **HMM:** predict when forward-filtered probability of the final state crosses
  its 2023-selected alarm threshold.
- **HMM ablation:** the same state model without normalized tyre age as an
  observation, to measure whether lap-degradation signals add value beyond
  tyre-age counting.

Pits are matched one-to-one within each driver-race at the selected lap
tolerance. Precision, recall, and F1 are pooled across driver-races, with
sensitivity results at +/-1, +/-2, and +/-3 laps. Matched predicted-minus-actual
lap differences and unmatched prediction/pit records are saved for timing-bias
analysis. Per-driver-race metrics are retained as supplementary detail.
Results are written to `results/evaluation_summary.csv`,
`results/evaluation_pooled_summary.csv`, and
`results/prediction_timing_deltas.csv`. Validation grid scores are saved to
`results/validation_tuning.csv`. Models and scalers are under `results/models/`;
plots are under `results/plots/`.

## Project structure

```text
f1-pitstop-hmm/
├── main.py                  # Single CLI entry point
├── src/
│   ├── data_loader.py       # FastF1 data fetching and cache
│   ├── features.py          # Filtering, feature engineering, stint grouping
│   ├── hmm.py               # Gaussian HMM, Baum-Welch, forward filtering
│   └── pipeline.py          # Season split, training, baselines, evaluation
├── tests/
│   └── test_pipeline.py     # Regression tests for grouping, filtering, and HMM
├── data/                    # Processed CSV (gitignored)
├── results/
│   ├── models/              # Trained models and scalers (gitignored)
│   └── plots/               # Evaluation and transition plots
├── cache/                   # FastF1 cache (gitignored)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/AceAryan/f1-pitstop-hmm.git
cd f1-pitstop-hmm
pip install -r requirements.txt
python main.py --help
```

FastF1 caches downloaded sessions locally. The explicit preprocessing command
downloads the selected race calendars; subsequent runs reuse the cache.

## CLI options

| Option | Description |
|---|---|
| `--preprocess` | Fetch data and create `data/processed.csv` |
| `--tune` | Tune thresholds and state count on 2023; fit candidates on 2020–2022 |
| `--train` | Train before `--test_year`; for 2024, refit with the selected 2023 settings |
| `--evaluate` | Evaluate the held-out `--test_year`; the 2024 final test runs once |
| `--year` | Season to use with preprocessing |
| `--years` | Complete seasons to use with preprocessing |
| `--gp` | Grand Prix names to use with preprocessing |
| `--test_year` | Held-out season; defaults to the latest available season |
| `--compounds` | HMM compounds (default: `SOFT MEDIUM HARD`) |
| `--n_iter` | Baum-Welch iterations (default: 50) |
| `--n_states` | States for non-final training (2, 3, or 4) |
| `--tolerance` | Pit matching window in laps (default: 2) |
| `--drivers` | Driver codes to evaluate |
| `--plot_drivers` | Driver codes for state plots |
