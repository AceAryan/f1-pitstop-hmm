"""Make PLACEHOLDER website data in the same formats as the pipeline outputs.

Lap times and pit-stop laps are real (Kaggle/Ergast CSVs in data/kaggle/).
Every model output (HMM beliefs, alarms, logistic positives, metrics) is FAKE:
it comes from a simple made-up rule with random noise, not from any trained model.
The files are marked so the website shows a "placeholder data" notice.

Run from the repo root:  python docs/tools/make_placeholder_data.py
Replace with real outputs: bash docs/tools/update_site_data.sh
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(335)
SRC = 'data/kaggle'
OUT = 'docs/data'
HMM_GRID = [round(v / 10, 1) for v in range(3, 10)]
AGE_GRID = [round(v / 100, 2) for v in range(60, 100, 5)]
PLACEHOLDER_NOTE = ('PLACEHOLDER: lap times and pit laps are real Ergast data; all model '
                    'outputs and metrics are fake, made by docs/tools/make_placeholder_data.py')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load(year):
    races = pd.read_csv(f'{SRC}/races.csv')
    races = races[races.year == year]
    laps = pd.read_csv(f'{SRC}/lap_times.csv')
    pits = pd.read_csv(f'{SRC}/pit_stops.csv')
    out = []
    for _, race in races.sort_values('round').iterrows():
        rl = laps[laps.raceId == race.raceId]
        rp = pits[pits.raceId == race.raceId]
        for driver_id, dl in rl.groupby('driverId'):
            dl = dl.sort_values('lap')
            stops = sorted(rp[rp.driverId == driver_id].lap.astype(int).tolist())
            out.append((race['name'], str(driver_id), dl, stops))
    return out


def simulate(dl, stops):
    """Fake, causal per-lap model outputs for one driver-race."""
    lap_numbers = dl.lap.astype(int).to_numpy()
    times = dl.milliseconds.to_numpy() / 1000.0
    median = np.median(times)
    stints = np.searchsorted(stops, lap_numbers, side='left') + 1
    rows = []
    state = {}
    for lap, t, stint in zip(lap_numbers, times, stints):
        if state.get('stint') != stint:
            state = {'stint': stint, 'kept': 0, 'best': None, 'deltas': [],
                     'p': None, 'p_na': None, 'pc': 0.0, 'pc_na': 0.0, 'log_prev': False}
        observed = (lap > 1 and lap not in stops and (lap - 1) not in stops
                    and t <= 1.35 * median)
        row = {'lapNumber': int(lap), 'lapTime': round(float(t), 3), 'stint': int(stint),
               'compound': 'UNKNOWN', 'modelObservation': bool(observed),
               'hmmStateProbabilities': state['p'], 'hmmNoAgeStateProbabilities': state['p_na'],
               'hmmAlarm': False, 'hmmNoAgeAlarm': False, 'tyreAgeAlarm': False,
               'logisticAlarm': False, 'logisticPositive': False, 'tyreAgeNorm': None,
               'actualPit': bool(lap in stops)}
        if observed:
            state['kept'] += 1
            state['best'] = t if state['best'] is None else min(state['best'], t)
            state['deltas'].append(t - state['best'])
            roll = float(np.mean(state['deltas'][-3:]))
            age = state['kept'] / 40
            raw = sigmoid(9 * (age - 0.62) + 0.6 * roll + RNG.normal(0, 0.6))
            state['pc'] = max(state['pc'] * 0.96, raw)
            raw_na = sigmoid(1.2 * (roll - 1.6) + 2.5 * (age - 0.5) + RNG.normal(0, 0.9))
            state['pc_na'] = max(state['pc_na'] * 0.93, raw_na)
            for key, pc_key in (('p', 'pc'), ('p_na', 'pc_na')):
                pc = state[pc_key]
                fresh = (1 - pc) * sigmoid(4 - 9 * age)
                state[key] = [round(fresh, 4), round(1 - pc - fresh, 4), round(pc, 4)]
            row['hmmStateProbabilities'] = state['p']
            row['hmmNoAgeStateProbabilities'] = state['p_na']
            row['tyreAgeNorm'] = round(age, 4)
            score = 7 * (age - 0.7) + 0.4 * roll + RNG.normal(0, 1.2)
            row['logisticPositive'] = bool(score > 0)
            if row['logisticPositive'] and not state['log_prev']:
                row['logisticAlarm'] = True
            state['log_prev'] = row['logisticPositive']
        rows.append(row)
    return rows


def first_entries(rows, value_fn, threshold):
    """One alarm per stint: the first observed lap where value >= threshold."""
    alarms, done = [], set()
    for r in rows:
        if not r['modelObservation'] or r['stint'] in done:
            continue
        v = value_fn(r)
        if v is not None and v >= threshold:
            alarms.append(r['lapNumber'])
            done.add(r['stint'])
    return alarms


def match(pred, actual, tol):
    if not pred or not actual:
        return []
    p, a = np.asarray(pred), np.asarray(actual)
    d = np.abs(p[:, None] - a[None, :])
    unmatched = (int(d.max()) + 1) * (min(len(p), len(a)) + 1)
    cost = np.full((len(p), len(a) + len(p)), unmatched, dtype=float)
    cost[:, :len(a)] = d
    if tol is not None:
        cost[:, :len(a)][d > tol] = unmatched * 2
    rr, cc = linear_sum_assignment(cost)
    return [(int(p[r]), int(a[c])) for r, c in zip(rr, cc)
            if c < len(a) and (tol is None or d[r, c] <= tol)]


def pooled(preds, actuals, tol):
    tp = fp = fn = 0
    for key in set(preds) | set(actuals):
        pr, ac = preds.get(key, []), actuals.get(key, [])
        m = len(match(pr, ac, tol))
        tp, fp, fn = tp + m, fp + len(pr) - m, fn + len(ac) - m
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': round(p, 3),
            'Recall': round(r, 3), 'F1': round(f, 3)}


METHOD_FNS = {
    'HMM': lambda r: (r['hmmStateProbabilities'] or [None])[-1],
    'HMMNoAge': lambda r: (r['hmmNoAgeStateProbabilities'] or [None])[-1],
    'TyreAge': lambda r: r['tyreAgeNorm'],
}


def predictions(sims, method, threshold):
    if method == 'Logistic':
        return {k: [r['lapNumber'] for r in rows if r['logisticAlarm']] for k, (rows, _) in sims.items()}
    return {k: first_entries(rows, METHOD_FNS[method], threshold) for k, (rows, _) in sims.items()}


def write_json(name, payload):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))


def main():
    os.makedirs(OUT, exist_ok=True)
    selected = {'hmm': 0.6, 'noAge': 0.7, 'age': 0.8}

    # ---- 2019 validation grid (fake) ----
    sims19 = {(gp, drv): (simulate(dl, stops), stops) for gp, drv, dl, stops in load(2019)}
    act19 = {k: v[1] for k, v in sims19.items()}

    def grid(method, thresholds):
        out = []
        for th in thresholds:
            m = pooled(predictions(sims19, method, th), act19, 2)
            out.append({'threshold': th, 'precision': m['Precision'], 'recall': m['Recall'],
                        'f1': m['F1'], 'tp': m['TP'], 'fp': m['FP'], 'fn': m['FN']})
        return out
    methods = {
        'HMM': {'nStates': 3, 'thresholds': grid('HMM', HMM_GRID)},
        'HMMNoAge': {'nStates': 3, 'thresholds': grid('HMMNoAge', HMM_GRID)},
        'TyreAge': {'thresholds': grid('TyreAge', AGE_GRID)},
    }
    for key in methods:  # select by best F1, as --tune does
        best = max(methods[key]['thresholds'], key=lambda r: r['f1'])
        methods[key]['selectedThreshold'] = best['threshold']
    selected = {'hmm': methods['HMM']['selectedThreshold'],
                'noAge': methods['HMMNoAge']['selectedThreshold'],
                'age': methods['TyreAge']['selectedThreshold']}
    lg = pooled(predictions(sims19, 'Logistic', None), act19, 2)
    methods['Logistic'] = {'precision': lg['Precision'], 'recall': lg['Recall'], 'f1': lg['F1'],
                           'tp': lg['TP'], 'fp': lg['FP'], 'fn': lg['FN']}
    write_json('validation-threshold-metrics-2019.json', {
        'schemaVersion': 1, 'placeholder': True, 'placeholderNote': PLACEHOLDER_NOTE,
        'validationYear': 2019, 'trainingYears': list(range(2011, 2019)),
        'matchingTolerance': 2, 'metricAveraging': 'pooled', 'methods': methods})

    # ---- 2020 test (fake) ----
    races = {}
    preds = {m: {} for m in ('HMM', 'HMMNoAge', 'TyreAge', 'Logistic')}
    actual = {}
    for gp, drv, dl, stops in load(2020):
        rows = simulate(dl, stops)
        key = (gp, drv)
        actual[key] = stops
        for method, th, flag in (('HMM', selected['hmm'], 'hmmAlarm'),
                                 ('HMMNoAge', selected['noAge'], 'hmmNoAgeAlarm'),
                                 ('TyreAge', selected['age'], 'tyreAgeAlarm')):
            laps = first_entries(rows, METHOD_FNS[method], th)
            preds[method][key] = laps
            for r in rows:
                if r['lapNumber'] in laps:
                    r[flag] = True
        preds['Logistic'][key] = [r['lapNumber'] for r in rows if r['logisticAlarm']]
        races.setdefault(gp, []).append({'driver': drv, 'actualPitLaps': stops, 'laps': rows})
    write_json('race-explorer-2020.json', {
        'schemaVersion': 1, 'placeholder': True, 'placeholderNote': PLACEHOLDER_NOTE,
        'year': 2020,
        'stateNames': {'0': 'FRESH', 'intermediate': 'DEGRADING', 'final': 'CRITICAL'},
        'alarmThresholds': {'hmm': selected['hmm'], 'tyreAge': selected['age'],
                            'logistic': 'classifier decision boundary'},
        'races': [{'year': 2020, 'grandPrix': gp, 'drivers': sorted(d, key=lambda x: int(x['driver']))}
                  for gp, d in races.items()]})

    pooled_rows, timing_rows = [], []
    for method, mp in preds.items():
        keys = sorted(set(mp) | set(actual))
        for tol in (1, 2, 3):
            pooled_rows.append({'Year': 2020, 'Method': method, 'Tolerance': tol,
                                'RaceDriverCount': len(keys), **pooled(mp, actual, tol)})
        for gp, drv in keys:
            pr, ac = list(mp.get((gp, drv), [])), list(actual.get((gp, drv), []))
            base = {'Year': 2020, 'GrandPrix': gp, 'Driver': drv, 'Method': method}
            for p, a in match(pr, ac, None):
                pr.remove(p)
                ac.remove(a)
                timing_rows.append({**base, 'PredictionLap': p, 'ActualPitLap': a,
                                    'PredictedMinusActual': p - a, 'MatchStatus': 'matched'})
            timing_rows += [{**base, 'PredictionLap': p, 'MatchStatus': 'unmatched_prediction'} for p in pr]
            timing_rows += [{**base, 'ActualPitLap': a, 'MatchStatus': 'unmatched_actual'} for a in ac]
    pd.DataFrame(pooled_rows).to_csv(os.path.join(OUT, 'evaluation_pooled_summary.csv'), index=False)
    cols = ['Year', 'GrandPrix', 'Driver', 'Method', 'PredictionLap', 'ActualPitLap',
            'PredictedMinusActual', 'MatchStatus']
    pd.DataFrame(timing_rows, columns=cols).to_csv(
        os.path.join(OUT, 'prediction_timing_deltas.csv'), index=False, float_format='%.0f')

    with open(os.path.join(OUT, 'placeholder.json'), 'w') as f:
        json.dump({'placeholder': True, 'note': PLACEHOLDER_NOTE,
                   'files': ['evaluation_pooled_summary.csv', 'prediction_timing_deltas.csv',
                             'race-explorer-2020.json', 'validation-threshold-metrics-2019.json']},
                  f, indent=2)
    print('selected (fake):', selected)
    print(pd.DataFrame(pooled_rows).query('Tolerance == 2')[['Method', 'TP', 'FP', 'FN', 'F1']])


if __name__ == '__main__':
    main()
