# F1 Pit-Stop HMM

Can lap times alone tell us when a Formula 1 driver is about to pit?

We treat tyre wear as a hidden state, infer it from lap times with a hidden
Markov model (HMM) written from scratch, and compare its pit-stop alarms with
two simple baselines.

**Project website:** https://acearyan.github.io/f1-pitstop-hmm/
(full write-up, interactive results and a race explorer)

ES 335 Machine Learning, IIT Gandhinagar, 2026.

## Result in one line

No model predicts the pit lap well. On the 2020 test season, at a ±2 lap window:

| Model | F1 |
|---|---|
| Tyre-age rule | 0.107 |
| HMM | 0.106 |
| HMM without tyre age | 0.101 |
| Logistic regression | 0.085 |

The HMM does not beat simply counting laps. See the website for why.

## Quick start

The Kaggle CSVs are already in the repo, so you can run everything straight
after cloning.

```bash
git clone https://github.com/AceAryan/f1-pitstop-hmm.git
cd f1-pitstop-hmm
pip install -r requirements.txt

python main.py --preprocess                   # clean laps, build features, split by season
python main.py --tune                         # fit on 2011-2018, choose settings on 2019
python main.py --train --test_year 2020       # refit on 2011-2019 with the 2019 settings
python main.py --evaluate --test_year 2020    # one-time final test on 2020
```

The 2020 test can only run once. The pipeline refuses to repeat it.

## Data

- Source: [Kaggle Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
  (Ergast data, CC0 Public Domain).
- Files used: `lap_times.csv`, `pit_stops.csv`, `races.csv`, `results.csv`
  in `data/kaggle/`.
- Optional: `tyres.csv` adds compound and tyre-life data. Without it, the
  compound is `UNKNOWN` and tyre age is counted from the stint.
- Seasons: 2011-2020.

| Seasons | Role |
|---|---|
| 2011-2018 | Training |
| 2019 | Validation (all tuning happens here) |
| 2020 | Final test, run once |

## How it works

### Cleaning

- Drop the first lap of each race.
- Drop pit-in and pit-out laps.
- Drop laps slower than 1.35x the driver's median lap in that race.
- Drop whole races that used intermediate or wet tyres.
- Infer stints from pit-stop laps. Laps are grouped by year, Grand Prix,
  driver and stint.

### Features (three per lap)

Each feature uses only the current stint, up to the current lap.

1. Lap-time delta: lap time minus the best lap so far in the stint.
2. Rolling degradation: 3-lap rolling mean of the delta.
3. Tyre age: laps so far in the stint, scaled by a typical tyre life.

Ergast has no sector times, so there are no sector features.

### Models

| Model | Raises an alarm when |
|---|---|
| HMM | P(final state), from forward filtering, crosses a threshold. Once per stint. |
| HMM without tyre age | Same, but tyre age is removed from the inputs (ablation). |
| Tyre-age rule | Tyre age crosses a threshold. Once per stint. |
| Logistic regression | It predicts a pit within the next 3 laps. Can fire more than once per stint. |

HMM details:

- Gaussian emissions, left-to-right transitions, always starts in the first state.
- Uses forward filtering, `P(state_t | laps up to t)`, not Viterbi, so it
  never looks at future laps.
- Number of states (2, 3 or 4) is chosen on 2019. The final model uses
  **2 states** (fresh, worn).

### What is tuned on 2019

- HMM state count: 2, 3 or 4.
- HMM alarm threshold: 0.3 to 0.9.
- Tyre-age threshold: 0.60 to 0.95.
- Each picked by pooled F1 at ±2 laps. The ablation is tuned the same way.

### Evaluation

- Alarms are matched one-to-one to real pit stops within each driver-race.
- A match within ±2 laps is a hit. Results are also reported at ±1 and ±3.
- Precision, recall and F1 are pooled over all driver-races.
- Alarm timing (alarm lap minus pit lap) is saved to check for early or late alarms.

## Outputs

All outputs go to `results/`, which is not committed.

| File | Contents |
|---|---|
| `evaluation_pooled_summary.csv` | Main results, pooled, at ±1, ±2, ±3 |
| `evaluation_summary.csv` | Per driver-race results |
| `prediction_timing_deltas.csv` | Alarm lap minus pit lap, plus unmatched alarms and pits |
| `validation_tuning.csv` | Every 2019 tuning score |
| `site-data/validation-threshold-metrics-2019.json` | 2019 threshold curves, for the website |
| `site-data/race-explorer-2020.json` | Lap-by-lap 2020 data, for the website |
| `models/` | Trained models and scalers |
| `plots/` | State plots |

To update the website with new results:

```bash
bash docs/tools/update_site_data.sh
```

## Command-line options

| Option | Description |
|---|---|
| `--preprocess` | Build train / validation / test splits from the Kaggle CSVs |
| `--tune` | Fit on 2011-2018 and tune on 2019 |
| `--train` | Train on seasons before `--test_year`. For 2020, uses the 2019 settings |
| `--evaluate` | Evaluate `--test_year`. The 2020 test runs only once |
| `--test_year` | Held-out season (default: 2020) |
| `--compounds` | Compounds to train on (default: `SOFT MEDIUM HARD`) |
| `--n_iter` | Baum-Welch iterations (default: 50) |
| `--n_states` | States for non-final training: 2, 3 or 4 (default: 3) |
| `--tolerance` | Pit matching window in laps (default: 2) |
| `--drivers` | Driver codes to evaluate |
| `--plot_drivers` | Driver codes to plot |

## Project structure

```text
f1-pitstop-hmm/
├── main.py                  # Command-line entry point
├── data/
│   ├── kaggle_loader.py     # Loads the Kaggle CSVs
│   ├── kaggle/              # Kaggle CSVs (committed)
│   └── processed/           # Train / validation / test splits
├── src/
│   ├── data_loader.py       # Older FastF1 loader
│   ├── features.py          # Cleaning, features, stints
│   ├── hmm.py               # Gaussian HMM, Baum-Welch, forward filtering
│   └── pipeline.py          # Split, training, baselines, evaluation
├── tests/
│   └── test_pipeline.py     # Unit tests
├── docs/                    # Project website (GitHub Pages)
└── requirements.txt
```

## Team

- **Aryan Kumar** (24110055): models and pipeline.
- **D A S K R Manognya** (24110097): website and presentation.
