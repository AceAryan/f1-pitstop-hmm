# 🏎️ f1-pitstop-hmm

> Inferring F1 tyre degradation states and predicting optimal pit stops using Hidden Markov Models.

---

## Overview

This project models tyre degradation in Formula 1 races as a Hidden Markov Model (HMM).
The hidden states represent tyre condition (`FRESH`, `DEGRADING`, `CRITICAL`), while
observations are lap-by-lap telemetry features (lap time delta, tyre age, sector times).

Key design choices:
- **One HMM per tyre compound** (Soft/Medium/Hard) — each compound degrades differently
- **Baum-Welch (EM)** to learn transition and emission parameters from historical stints
- **Viterbi algorithm** to decode the most likely tyre state sequence per race
- **State smoothing** — require N consecutive CRITICAL laps before predicting a pit stop (reduces false positives)

---

## Pipeline

```
python main.py                          # full pipeline (preprocess → train → evaluate)
python main.py --preprocess             # only load data + engineer features
python main.py --train                  # only train HMMs
python main.py --evaluate               # only evaluate + plot
python main.py --preprocess --train     # chain specific steps
```

---

## Hidden States

| State | Meaning |
|---|---|
| `0 — FRESH` | Tyre performing at peak |
| `1 — DEGRADING` | Performance dropping, lap times rising |
| `2 — CRITICAL` | Tyre worn out, pit stop imminent |

---

## Features Used

| Feature | Description |
|---|---|
| `lap_time_delta` | Lap time vs driver's session-best in that stint |
| `tyre_age_norm` | Laps on compound, normalized by compound max life |
| `sector3_delta` | Sector 3 time vs best (most sensitive to rear tyre wear) |
| `rolling_degradation` | 3-lap rolling mean of lap_time_delta (captures trend) |

---

## Project Structure

```
f1-pitstop-hmm/
├── main.py                  # Single entry point (--preprocess / --train / --evaluate)
├── src/
│   ├── data_loader.py       # FastF1 data fetching & caching
│   ├── features.py          # Feature engineering + compound grouping
│   ├── hmm.py               # HMM from scratch (Forward, Viterbi, Baum-Welch)
│   └── eval_utils.py        # Plot utilities
├── data/                    # Processed parquet files (gitignored)
├── results/
│   ├── models/              # Trained HMM pickles (gitignored)
│   └── plots/               # State sequence + transition matrix plots
├── cache/                   # FastF1 cache (gitignored, can be large)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/yourname/f1-pitstop-hmm.git
cd f1-pitstop-hmm
pip install -r requirements.txt
python main.py
```

---

## CLI Options

All steps live in `main.py`. Pass step flags to run only what you need.

| Flag | Description |
|---|---|
| `--preprocess` | Run preprocessing step only |
| `--train` | Run training step only |
| `--evaluate` | Run evaluation step only |
| *(none)* | Run all three steps |

### Preprocess
| Flag | Default | Description |
|---|---|---|
| `--year` | 2022+2023 | Season year (2018–2024) |
| `--gp` | 6 default GPs | Grand Prix names |

### Train
| Flag | Default | Description |
|---|---|---|
| `--compounds` | SOFT MEDIUM HARD | Compounds to train |
| `--n_iter` | 50 | Baum-Welch EM iterations |
| `--n_states` | 3 | Number of hidden states |

### Evaluate
| Flag | Default | Description |
|---|---|---|
| `--smooth` | 3 | Consecutive CRITICAL laps before predicting pit |
| `--tolerance` | 2 | Allowed lap window for TP match |
| `--drivers` | top 10 | Driver codes to evaluate |
| `--plot_drivers` | top 3 | Drivers to save state plots for |

---

## Team

| Name | Contribution |
|---|---|
| Teammate A | Data pipeline, feature engineering, visualizations |
| Teammate B | HMM derivations, implementation, evaluation |

---

## References

- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.*
- FastF1 Python library: https://theoehrly.github.io/Fast-F1/
- Ergast F1 API: http://ergast.com/mrd/