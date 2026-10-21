# F1 Pit-Stop HMM

Infer tyre-degradation states from Formula 1 lap data and compare pit-stop
predictions against two baselines: a tyre-age threshold and logistic regression.

## Pipeline

The single entry point is `main.py`; processed observations use CSV for the
existing training/evaluation pipeline and pickle files for the season splits.

```text
python main.py --preprocess
python main.py --tune                           # fit on 2011-2018; select using 2019 only
python main.py --train --test_year 2020         # refit on 2011-2019 using selected settings
python main.py --evaluate --test_year 2020      # one-time final test
```

The final experiment is chronological: candidate models are fit on 2011–2018,
all threshold and state-count selection uses 2019, and final models are refit
on 2011–2019 before the 2020 test is evaluated once. `--tune` selects the
terminal-state posterior threshold (0.3–0.9), tyre-age threshold (0.60–0.95),
and state count (2, 3, or 4) by pooled 2019 F1 at +/-2 laps. It also tunes the
no-tyre-age HMM ablation on the same validation season. Fitted scalers and
models use training seasons only. The 2020 final evaluation is guarded against
repeat runs; it requires final models trained with the saved 2019 configuration.
Preprocessing writes `data/processed/train.pkl`, `val.pkl`, and `test.pkl`, as
well as the combined CSV used by training and evaluation. It also preserves
complete dry-race lap records in `data/race_laps.csv` for the race explorer.

Ergast observations are grouped by **Year, Grand Prix, Driver, and inferred
stint**. The pipeline retains real lap numbers and pit targets, excludes
pit-in/out laps and the first race lap, and removes lap times over 1.35 times
the driver's median in that race. Entire races with known intermediate or wet
compounds are excluded. When compound data is unavailable, compound is marked
`UNKNOWN` and tyre age is inferred from stint lap counts. Lap-time delta and
rolling degradation use only lap times from the current stint. Ergast has no
sector times, so the feature vector contains three features. Regenerate
processed data with `--preprocess` after changing the source CSVs.

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

- **Tyre-age rule:** predict once per stint at the threshold selected on 2019.
- **Logistic regression:** predict within a three-lap horizon, trained using
  training-season pit targets.
- **HMM:** predict when forward-filtered probability of the final state crosses
  its 2019-selected alarm threshold.
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
`results/validation_tuning.csv`. `--tune` also exports
`results/site-data/validation-threshold-metrics-2019.json`, containing the
Python-computed pooled metric curve for each tuned threshold. A successful
`--evaluate --test_year 2020` exports
`results/site-data/race-explorer-2020.json`, with race/driver timelines,
per-lap forward-filtered state probabilities, available lap times,
HMM and baseline alarm flags, model-observation markers, and actual pit laps.
On laps excluded from model observations, probabilities carry forward from the
latest observation in that stint; the model-observation marker distinguishes
new updates from carried values. These JSON files are intended as static
GitHub Pages inputs; no website runtime computation is required. Models and
scalers are under `results/models/`; plots are under `results/plots/`.

## Project structure

```text
f1-pitstop-hmm/
├── main.py                  # Single CLI entry point
├── data/
│   ├── kaggle_loader.py     # Kaggle Ergast CSV loader
│   ├── kaggle/              # Downloaded source CSVs (gitignored)
│   └── processed/           # Train/validation/test pickle splits
├── src/
│   ├── features.py          # Filtering, feature engineering, stint grouping
│   ├── hmm.py               # Gaussian HMM, Baum-Welch, forward filtering
│   └── pipeline.py          # Season split, training, baselines, evaluation
├── tests/
│   └── test_pipeline.py     # Regression tests for grouping, filtering, and HMM
├── results/
│   ├── models/              # Trained models and scalers (gitignored)
│   └── plots/               # Evaluation and transition plots
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/AceAryan/f1-pitstop-hmm.git
cd f1-pitstop-hmm
pip install -r requirements.txt
python main.py --help
```

Download the [Kaggle Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020),
unzip the CSV files into `data/kaggle/`, then run:

```bash
python main.py --preprocess
```

The required files are `lap_times.csv`, `pit_stops.csv`, `races.csv`, and
`results.csv`. If available, `tyres.csv` provides compound and tyre-life
metadata; otherwise compounds are marked `UNKNOWN` and tyre age is inferred.

## CLI options

| Option | Description |
|---|---|
| `--preprocess` | Process Kaggle CSVs into train/validation/test splits |
| `--tune` | Tune thresholds and state count on 2019; fit candidates on 2011–2018 |
| `--train` | Train before `--test_year`; for 2020, refit with selected 2019 settings |
| `--evaluate` | Evaluate the held-out `--test_year`; the 2020 final test runs once |
| `--test_year` | Held-out season (default: 2020) |
| `--compounds` | HMM compounds (default: `SOFT MEDIUM HARD`) |
| `--n_iter` | Baum-Welch iterations (default: 50) |
| `--n_states` | States for non-final training (2, 3, or 4) |
| `--tolerance` | Pit matching window in laps (default: 2) |
| `--drivers` | Driver codes to evaluate |
| `--plot_drivers` | Driver codes for state plots |
