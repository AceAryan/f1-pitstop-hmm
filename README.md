# F1 Pit-Stop HMM

Infer tyre-degradation states from Formula 1 lap data and compare pit-stop
predictions against two baselines: a tyre-age threshold and logistic regression.

## Pipeline

The single entry point is `main.py`; processed observations use CSV throughout.

```text
python main.py                                  # preprocess, train, and evaluate
python main.py --preprocess                     # fetch laps and engineer features
python main.py --train                          # train on seasons before held-out season
python main.py --evaluate                       # evaluate the held-out season
python main.py --preprocess --train
python main.py --preprocess --year 2022 --gp Bahrain Monaco
python main.py --preprocess --years 2020 2021 2022 2023
python main.py --train --test_year 2023 --n_iter 80
python main.py --evaluate --test_year 2023 --drivers VER HAM LEC
```

Preprocessing defaults to dry-tyre races in seasons 2020–2023. The latest
season in `data/processed.csv` (2023 by default) is held out; this trains on
2020–2022 and evaluates on 2023. The 2024 season can be added as a held-out
test set if API access allows. Training only uses earlier seasons; the feature
scaler and logistic-regression baseline are fit on training data only. At
least two seasons are required, and a requested test season must have an
earlier training season. Use `--years` to choose complete seasons or
`--year`/`--gp` to select one season or specific races.

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
| 1 | DEGRADING |
| 2 | CRITICAL |

The HMM starts with `pi = [1, 0, 0]`, uses left-to-right state transitions, and
initializes its state emissions in increasing tyre-age order. Evaluation uses
forward-filtered probabilities (`P(state_t | laps through t)`), not Viterbi
decoding, so a prediction never uses future laps from the same stint.

## Baselines and output

- **Tyre-age rule:** predict once per stint when normalized tyre age reaches 0.8.
- **Logistic regression:** predict within a three-lap horizon, trained using
  training-season pit targets.
- **HMM:** predict when forward-filtered states first enter CRITICAL.

Pit predictions are matched to real pit-in `LapNumber` values using the
`--tolerance` lap window. Results are written to
`results/evaluation_summary.csv`, with separate metrics for all three methods.
Trained compound HMMs, fitted scalers, and the baseline are stored under
`results/models/`; evaluation plots are saved under `results/plots/`.

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
python main.py
```

FastF1 caches downloaded sessions locally. The default run downloads the
complete 2020–2023 race calendars, so the initial preprocessing run can take a
while; subsequent runs reuse the cache.

## CLI options

| Option | Description |
|---|---|
| `--preprocess` | Fetch data and create `data/processed.csv` |
| `--train` | Train on seasons earlier than `--test_year` |
| `--evaluate` | Evaluate only the held-out `--test_year` |
| `--year` | Season to use with preprocessing |
| `--years` | Complete seasons to use with preprocessing |
| `--gp` | Grand Prix names to use with preprocessing |
| `--test_year` | Held-out season; defaults to the latest available season |
| `--compounds` | HMM compounds (default: `SOFT MEDIUM HARD`) |
| `--n_iter` | Baum-Welch iterations (default: 50) |
| `--n_states` | Number of states; must be 3 |
| `--tolerance` | Pit matching window in laps (default: 2) |
| `--drivers` | Driver codes to evaluate |
| `--plot_drivers` | Driver codes for state plots |
