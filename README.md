# f1-pitstop-hmm

> Inferring F1 tyre degradation states and predicting optimal pit stops using Hidden Markov Models.

---

## Overview

This project models tyre degradation in Formula 1 races as a Hidden Markov Model (HMM).
The hidden states represent tyre condition (`FRESH`, `DEGRADING`, `CRITICAL`), while
observations are lap-by-lap telemetry features like lap time delta and tyre age.

We use the **Viterbi algorithm** to decode the most likely tyre state sequence per race,
and **Baum-Welch (EM)** to learn transition and emission parameters from historical data.

---

## Pipeline

```
Raw Telemetry (FastF1)
        ↓
Feature Engineering
        ↓
HMM Training (Baum-Welch)
        ↓
Viterbi Decoding → Tyre State Sequence
        ↓
Pit Stop Prediction & Evaluation
```

---

## Hidden States

| State | Meaning |
|---|---|
| `0 - FRESH` | Tyre performing at peak |
| `1 - DEGRADING` | Performance dropping, lap times rising |
| `2 - CRITICAL` | Tyre worn out, pit stop imminent |

---

## Features Used

| Feature | Description |
|---|---|
| `lap_time_delta` | Lap time vs driver's session best |
| `tyre_age` | Laps completed on current compound |
| `sector3_delta` | Sector 3 time vs best (rear tyre sensitive) |
| `gap_delta` | Change in gap to car ahead |

---

## Project Structure

```
f1-pitstop-hmm/
├── src/
│   ├── data_loader.py       # FastF1 data fetching & caching
│   ├── features.py          # Feature engineering
│   ├── hmm.py               # HMM from scratch (forward, Viterbi, Baum-Welch)
│   └── evaluate.py          # Evaluation metrics
├── notebooks/
│   └── exploration.ipynb    # EDA and visualizations
├── results/                 # Saved outputs, plots
├── cache/                   # FastF1 cache
├── main.py                  # Entry point
├── requirements.txt
└── README.md
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

## Team

| Name | Contribution |
|---|---|
| Aryan Kumar | Data pipeline, feature engineering, visualizations |
| DASKR Manognya | HMM derivations, implementation, evaluation |

---

## References

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.*
- FastF1 Python library: https://theoehrly.github.io/Fast-F1/
- Ergast F1 API: http://ergast.com/mrd/