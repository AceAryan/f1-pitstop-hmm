/* ES 335 F1 pit-stop HMM website. Plain JS, no libraries.
 * Reads pipeline outputs from data/ (see README section "Rebuild this website"). */
(() => {
  'use strict';

  const DATA = 'data/';
  const METHODS = {
    HMM: { label: 'HMM', color: 'var(--hmm)', flag: 'hmmAlarm' },
    HMMNoAge: { label: 'HMM without tyre age', color: 'var(--noage)', flag: 'hmmNoAgeAlarm' },
    TyreAge: { label: 'Tyre-age rule', color: 'var(--age)', flag: 'tyreAgeAlarm' },
    Logistic: { label: 'Logistic regression', color: 'var(--logit)', flag: 'logisticAlarm' },
  };
  const ORDER = ['HMM', 'HMMNoAge', 'TyreAge', 'Logistic'];
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ---------- loading ---------- */
  async function getJSON(name, optional = false) {
    const res = await fetch(DATA + name, { cache: 'no-cache' });
    if (!res.ok) {
      if (optional) return null;
      throw new Error(`${name}: HTTP ${res.status}`);
    }
    return res.json();
  }
  async function getCSV(name) {
    const res = await fetch(DATA + name, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    return parseCSV(await res.text());
  }
  function parseCSV(text) {
    const rows = [];
    let row = [], field = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') q = false;
        else field += c;
      } else if (c === '"') q = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = '';
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else field += c;
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    const head = rows.shift();
    return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
  }
  const num = (v) => (v === '' || v == null ? null : Number(v));

  function showError(msg) {
    const el = $('load-error');
    el.hidden = false;
    el.innerHTML = msg;
  }

  /* ---------- SVG helpers ---------- */
  const NS = 'http://www.w3.org/2000/svg';
  function svgSize(svg, h) {
    const w = Math.max(280, Math.round(svg.parentElement.getBoundingClientRect().width));
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    return w;
  }
  const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  function ticks(min, max, n = 5) {
    const step0 = (max - min) / n;
    const mag = 10 ** Math.floor(Math.log10(step0 || 1));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function el(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  /* ---------- placeholder notice ---------- */
  function markPlaceholder() {
    $('placeholder-notice').hidden = false;
    for (const id of ['evaluation', 'results', 'explorer', 'errors']) {
      const h = document.querySelector(`#${id} h2`);
      if (h && !h.querySelector('.ph-tag')) {
        const tag = document.createElement('span');
        tag.className = 'ph-tag';
        tag.textContent = 'placeholder data';
        tag.style.cssText = 'font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#fff;background:var(--accent);padding:2px 8px;border-radius:999px;margin-left:10px;vertical-align:middle';
        h.appendChild(tag);
      }
    }
  }

  /* ================= THRESHOLD CONTROL (2019) ================= */
  function initThreshold(val) {
    const sel = $('thr-method'), slider = $('thr-slider');
    function current() { return val.methods[sel.value]; }
    function setMethod() {
      const m = current();
      slider.max = m.thresholds.length - 1;
      const idx = m.thresholds.findIndex((t) => Math.abs(t.threshold - m.selectedThreshold) < 1e-9);
      slider.value = idx >= 0 ? idx : 0;
      $('pr-sw').style.background = METHODS[sel.value].color;
      render();
    }
    function render() {
      const m = current();
      const t = m.thresholds[+slider.value];
      const isAge = sel.value === 'TyreAge';
      $('thr-value').textContent = isAge
        ? `${t.threshold.toFixed(2)} (≈ ${Math.round(t.threshold * 40)} laps)`
        : `P(worn) ≥ ${t.threshold.toFixed(1)}`;
      $('thr-p').textContent = fmt(t.precision);
      $('thr-r').textContent = fmt(t.recall);
      $('thr-f').textContent = fmt(t.f1);
      const chosen = Math.abs(t.threshold - m.selectedThreshold) < 1e-9;
      const lg = val.methods.Logistic;
      $('thr-text').innerHTML =
        `${t.tp + t.fp} alarms raised in ${val.validationYear}: <strong>${t.tp}</strong> within ±${val.matchingTolerance} laps of a real pit, ` +
        `<strong>${t.fp}</strong> false alarms, <strong>${t.fn}</strong> pits missed. ` +
        (chosen ? '<strong>This is the setting chosen for the 2020 test.</strong> ' : '') +
        (m.nStates ? `HMM with ${m.nStates} states. ` : '') +
        `<span class="muted">Logistic regression for comparison: P ${fmt(lg.precision)}, R ${fmt(lg.recall)}, F1 ${fmt(lg.f1)}.</span>`;
      drawPR(m, +slider.value, lg);
      drawF1(m, +slider.value, isAge);
    }
    function drawPR(m, idx, lg) {
      const svg = $('pr-chart');
      const H = 280, w = svgSize(svg, H);
      svg.replaceChildren();
      const pad = { l: 44, r: 14, t: 12, b: 38 };
      const pts = m.thresholds.map((t) => [t.recall, t.precision]).concat([[lg.recall, lg.precision]]);
      const xmax = Math.min(1, Math.max(...pts.map((p) => p[0])) * 1.15 + 0.02);
      const ymax = Math.min(1, Math.max(...pts.map((p) => p[1])) * 1.15 + 0.02);
      const x = linear(0, xmax, pad.l, w - pad.r), y = linear(0, ymax, H - pad.b, pad.t);
      axes(svg, x, y, [0, xmax], [0, ymax], w, H, pad, 'Recall', 'Precision');
      const color = METHODS[sel.value].color;
      svg.append(el('polyline', { points: m.thresholds.map((t) => `${x(t.recall)},${y(t.precision)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2, opacity: 0.6 }));
      m.thresholds.forEach((t, i) => {
        const c = el('circle', { cx: x(t.recall), cy: y(t.precision), r: i === idx ? 7 : 4.5, fill: color, style: 'cursor:pointer' });
        c.append(el('title', {}, `threshold ${t.threshold}: P ${fmt(t.precision)}, R ${fmt(t.recall)}, F1 ${fmt(t.f1)}`));
        c.addEventListener('click', () => { slider.value = i; render(); });
        svg.append(c);
        if (Math.abs(t.threshold - m.selectedThreshold) < 1e-9) {
          svg.append(el('circle', { cx: x(t.recall), cy: y(t.precision), r: 11, fill: 'none', stroke: 'var(--text)', 'stroke-width': 2 }));
        }
      });
      const lx = x(lg.recall), ly = y(lg.precision);
      svg.append(el('rect', { x: lx - 5, y: ly - 5, width: 10, height: 10, fill: 'var(--logit)', transform: `rotate(45 ${lx} ${ly})` }));
    }
    function drawF1(m, idx, isAge) {
      const svg = $('f1-chart');
      const H = 150, w = svgSize(svg, H);
      svg.replaceChildren();
      const pad = { l: 40, r: 10, t: 10, b: 32 };
      const n = m.thresholds.length;
      const ymax = Math.max(0.05, ...m.thresholds.map((t) => t.f1)) * 1.2;
      const bw = (w - pad.l - pad.r) / n;
      const y = linear(0, ymax, H - pad.b, pad.t);
      for (const v of ticks(0, ymax, 3)) {
        svg.append(el('line', { x1: pad.l, x2: w - pad.r, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-dasharray': '2 3' }));
        svg.append(el('text', { x: pad.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, v.toFixed(2)));
      }
      m.thresholds.forEach((t, i) => {
        const r = el('rect', { x: pad.l + i * bw + 3, y: y(t.f1), width: Math.max(2, bw - 6), height: y(0) - y(t.f1), fill: METHODS[sel.value].color, opacity: i === idx ? 1 : 0.35, style: 'cursor:pointer' });
        r.addEventListener('click', () => { slider.value = i; render(); });
        svg.append(r);
        svg.append(el('text', { x: pad.l + (i + 0.5) * bw, y: H - pad.b + 14, 'text-anchor': 'middle' }, isAge ? t.threshold.toFixed(2) : t.threshold.toFixed(1)));
      });
      svg.append(el('text', { x: pad.l + (w - pad.l - pad.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'label' }, 'F1 by threshold'));
    }
    sel.addEventListener('change', setMethod);
    slider.addEventListener('input', render);
    setMethod();
    return render;
  }

  function axes(svg, x, y, xd, yd, w, H, pad, xl, yl) {
    for (const v of ticks(yd[0], yd[1], 4)) {
      svg.append(el('line', { x1: pad.l, x2: w - pad.r, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-dasharray': '2 3' }));
      svg.append(el('text', { x: pad.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, String(+v.toFixed(3))));
    }
    for (const v of ticks(xd[0], xd[1], Math.max(3, Math.floor((w - pad.l) / 70)))) {
      svg.append(el('text', { x: x(v), y: H - pad.b + 15, 'text-anchor': 'middle' }, String(+v.toFixed(3))));
    }
    svg.append(el('line', { x1: pad.l, x2: w - pad.r, y1: H - pad.b, y2: H - pad.b, stroke: 'var(--muted)' }));
    if (xl) svg.append(el('text', { x: (pad.l + w - pad.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'label' }, xl));
    if (yl) svg.append(el('text', { x: 12, y: (pad.t + H - pad.b) / 2, 'text-anchor': 'middle', class: 'label', transform: `rotate(-90 12 ${(pad.t + H - pad.b) / 2})` }, yl));
  }

  /* ================= RESULTS (2020) ================= */
  function initResults(pooled) {
    const rows = pooled.map((r) => ({ ...r, Tolerance: +r.Tolerance, TP: +r.TP, FP: +r.FP, FN: +r.FN, Precision: +r.Precision, Recall: +r.Recall, F1: +r.F1, RaceDriverCount: +r.RaceDriverCount }));
    let tol = 2;
    const buttons = document.querySelectorAll('#results [data-tol]');
    function render() {
      buttons.forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.tol === tol)));
      const sel = ORDER.map((m) => rows.find((r) => r.Method === m && r.Tolerance === tol)).filter(Boolean);
      const best = Math.max(...sel.map((r) => r.F1));
      $('results-table').innerHTML = `<table><thead><tr><th>Model</th><th class="num">Alarms</th><th class="num">TP</th><th class="num">FP</th><th class="num">FN</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr></thead><tbody>${
        sel.map((r) => `<tr class="${r.F1 === best ? 'best' : ''}"><td><span class="sw" style="background:${METHODS[r.Method].color}"></span> ${METHODS[r.Method].label}</td><td class="num">${r.TP + r.FP}</td><td class="num">${r.TP}</td><td class="num">${r.FP}</td><td class="num">${r.FN}</td><td class="num">${fmt(r.Precision)}</td><td class="num">${fmt(r.Recall)}</td><td class="num">${fmt(r.F1)}</td></tr>`).join('')
      }</tbody></table><p class="small muted" style="padding:6px 10px;margin:0">Pooled over ${sel[0]?.RaceDriverCount ?? '–'} driver-races, ±${tol} laps. Bold = highest F1.</p>`;
      const svg = $('results-chart');
      const rowH = 30, H = sel.length * rowH + 30, w = svgSize(svg, H);
      svg.replaceChildren();
      const lab = Math.min(170, w * 0.42);
      const x = linear(0, Math.max(0.1, best * 1.15), lab, w - 40);
      sel.forEach((r, i) => {
        const yy = i * rowH + 6;
        svg.append(el('text', { x: lab - 8, y: yy + 15, 'text-anchor': 'end', style: 'fill:var(--text);font-size:12px' }, METHODS[r.Method].label));
        svg.append(el('rect', { x: lab, y: yy + 3, width: Math.max(1, x(r.F1) - lab), height: 18, rx: 3, fill: METHODS[r.Method].color }));
        svg.append(el('text', { x: x(r.F1) + 6, y: yy + 16, style: 'fill:var(--text);font-size:12px;font-weight:600' }, fmt(r.F1)));
      });
      svg.append(el('text', { x: lab, y: H - 4, class: 'label' }, `F1 at ±${tol} laps`));
    }
    buttons.forEach((b) => b.addEventListener('click', () => { tol = +b.dataset.tol; render(); }));
    render();
    return render;
  }

  /* ================= TIMING (errors) ================= */
  function initTiming(timing) {
    const sel = $('timing-method');
    const LIM = 10;
    function render() {
      const m = sel.value;
      const rows = timing.filter((r) => r.Method === m);
      const deltas = rows.filter((r) => r.MatchStatus === 'matched').map((r) => num(r.PredictedMinusActual)).filter((v) => v != null);
      const unPred = rows.filter((r) => r.MatchStatus === 'unmatched_prediction').length;
      const unPit = rows.filter((r) => r.MatchStatus === 'unmatched_actual').length;
      const bins = [];
      for (let d = -LIM - 1; d <= LIM + 1; d++) bins.push({ d, n: 0 });
      for (const v of deltas) bins[Math.max(-LIM - 1, Math.min(LIM + 1, v)) + LIM + 1].n++;
      const svg = $('timing-chart');
      const H = 220, w = svgSize(svg, H);
      svg.replaceChildren();
      const pad = { l: 40, r: 10, t: 12, b: 40 };
      const bw = (w - pad.l - pad.r) / bins.length;
      const ymax = Math.max(1, ...bins.map((b) => b.n)) * 1.1;
      const y = linear(0, ymax, H - pad.b, pad.t);
      const x0 = pad.l + (LIM + 1 - 2) * bw, x1 = pad.l + (LIM + 1 + 3) * bw;
      svg.append(el('rect', { x: x0, y: pad.t, width: x1 - x0, height: H - pad.b - pad.t, fill: 'var(--good)', opacity: 0.1 }));
      svg.append(el('text', { x: (x0 + x1) / 2, y: pad.t + 12, 'text-anchor': 'middle', style: 'fill:var(--good);font-size:11px;font-weight:600' }, '±2 = hit'));
      for (const v of ticks(0, ymax, 4)) {
        svg.append(el('line', { x1: pad.l, x2: w - pad.r, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-dasharray': '2 3' }));
        svg.append(el('text', { x: pad.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, String(v)));
      }
      const every = bw < 16 ? 5 : bw < 26 ? 2 : 1;
      bins.forEach((b, i) => {
        const r = el('rect', { x: pad.l + i * bw + 1, y: y(b.n), width: Math.max(1, bw - 2), height: y(0) - y(b.n), fill: METHODS[m].color, opacity: Math.abs(b.d) <= 2 ? 1 : 0.45 });
        r.append(el('title', {}, `${Math.abs(b.d) > LIM ? (b.d < 0 ? `≤ −${LIM + 1}` : `≥ +${LIM + 1}`) : (b.d > 0 ? '+' : '') + b.d}: ${b.n} alarms`));
        svg.append(r);
        if (Math.abs(b.d) > LIM || (b.d % every === 0 && !(every > 1 && Math.abs(b.d) === LIM))) {
          const label = b.d < -LIM ? `≤−${LIM + 1}` : b.d > LIM ? `≥${LIM + 1}` : (b.d > 0 ? '+' : '') + b.d;
          svg.append(el('text', { x: pad.l + (i + 0.5) * bw, y: H - pad.b + 14, 'text-anchor': 'middle' }, label));
        }
      });
      svg.append(el('text', { x: pad.l, y: H - 4, class: 'label' }, '← early'));
      svg.append(el('text', { x: w - pad.r, y: H - 4, class: 'label', 'text-anchor': 'end' }, 'late →'));
      svg.append(el('text', { x: (pad.l + w - pad.r) / 2, y: H - 4, 'text-anchor': 'middle' }, 'alarm lap − pit lap'));
      const c = (f) => deltas.filter(f).length;
      const stats = [
        [c((d) => d >= -2 && d <= -1), 'hits 1–2 laps early'],
        [c((d) => d === 0), 'hits on the pit lap'],
        [c((d) => d >= 1 && d <= 2), 'hits 1–2 laps late (after the stop)'],
        [unPred, 'alarms with no pit left to pair'],
        [unPit, 'pits with no alarm left to pair'],
      ];
      $('timing-stats').innerHTML = stats.map(([n, l]) => `<div class="card"><div class="stat">${n}</div><div class="stat-label">${l}</div></div>`).join('');
    }
    sel.addEventListener('change', render);
    render();
    return render;
  }

  /* ================= FINAL-STINT FALSE ALARMS ================= */
  function finalStintTable(explorer, pooled) {
    const counts = Object.fromEntries(ORDER.map((m) => [m, { inFinal: 0, falseFinal: 0, total: 0 }]));
    for (const race of explorer.races) {
      for (const d of race.drivers) {
        const lastStint = Math.max(...d.laps.map((l) => l.stint ?? 0));
        const pits = d.actualPitLaps || [];
        for (const l of d.laps) {
          for (const m of ORDER) {
            if (!l[METHODS[m].flag]) continue;
            counts[m].total++;
            if (l.stint === lastStint) {
              counts[m].inFinal++;
              if (!pits.some((p) => Math.abs(p - l.lapNumber) <= 2)) counts[m].falseFinal++;
            }
          }
        }
      }
    }
    const fpAt2 = Object.fromEntries(pooled.filter((r) => +r.Tolerance === 2).map((r) => [r.Method, +r.FP]));
    $('final-stint-table').innerHTML = `<table><thead><tr><th>Model</th><th class="num">All alarms</th><th class="num">In final stint</th><th class="num">…with no pit within ±2</th><th class="num">All false alarms (±2)</th><th class="num">Share from final stint</th></tr></thead><tbody>${
      ORDER.map((m) => {
        const c = counts[m], fp = fpAt2[m];
        return `<tr><td><span class="sw" style="background:${METHODS[m].color}"></span> ${METHODS[m].label}</td><td class="num">${c.total}</td><td class="num">${c.inFinal}</td><td class="num">${c.falseFinal}</td><td class="num">${fp ?? '–'}</td><td class="num">${fp ? Math.round((100 * c.falseFinal) / fp) + '%' : '–'}</td></tr>`;
      }).join('')
    }</tbody></table>`;
  }

  /* ================= RACE EXPLORER ================= */
  function initExplorer(explorer, drivers, val) {
    const raceSel = $('ex-race'), drvSel = $('ex-driver'), modelSel = $('ex-model'), lapIn = $('ex-lap');
    const name = (id) => (drivers[id] ? `${drivers[id].name} (${drivers[id].code})` : `Driver #${id}`);
    const thresholds = {
      hmm: explorer.alarmThresholds?.hmm ?? val?.methods?.HMM?.selectedThreshold,
      noage: val?.methods?.HMMNoAge?.selectedThreshold,
    };
    explorer.races.forEach((r, i) => raceSel.append(new Option(`${r.grandPrix}`, i)));
    let race, drv, laps, maxLap, timer = null;

    function fillDrivers() {
      race = explorer.races[+raceSel.value];
      const prev = drv?.driver;
      drvSel.replaceChildren();
      [...race.drivers].sort((a, b) => name(a.driver).localeCompare(name(b.driver)))
        .forEach((d) => drvSel.append(new Option(name(d.driver), d.driver)));
      if (prev && race.drivers.some((d) => d.driver === prev)) drvSel.value = prev;
      else if (race.drivers.some((d) => d.driver === '1')) drvSel.value = '1';
      setDriver(true);
    }
    function setDriver(reset) {
      drv = race.drivers.find((d) => d.driver === drvSel.value) || race.drivers[0];
      laps = drv.laps;
      maxLap = Math.max(...race.drivers.flatMap((d) => d.laps.map((l) => l.lapNumber)));
      lapIn.min = laps[0].lapNumber;
      lapIn.max = laps[laps.length - 1].lapNumber;
      if (reset) lapIn.value = lapIn.min;
      else lapIn.value = Math.min(+lapIn.value, +lapIn.max);
      render();
    }
    const probKey = () => (modelSel.value === 'hmm' ? 'hmmStateProbabilities' : 'hmmNoAgeStateProbabilities');
    function stateNames(n) {
      if (n === 2) return ['Fresh', 'Worn'];
      if (n === 3) return ['Fresh', 'Degrading', 'Worn'];
      return ['Fresh', ...Array.from({ length: n - 2 }, (_, i) => `Degrading ${i + 1}`), 'Worn'];
    }
    function stateColors(n) {
      if (n === 2) return ['var(--fresh)', 'var(--critical)'];
      const mids = Array.from({ length: n - 2 }, (_, i) => `color-mix(in srgb, var(--degrading) ${100 - i * 35}%, var(--critical))`);
      return ['var(--fresh)', ...mids, 'var(--critical)'];
    }

    function render() {
      const cur = +lapIn.value;
      const seen = laps.filter((l) => l.lapNumber <= cur);
      const now = laps.find((l) => l.lapNumber === cur) || seen[seen.length - 1];
      const key = probKey();
      const nStates = (laps.find((l) => l[key])?.[key] || [0, 0, 0]).length;
      const names = stateNames(nStates), colors = stateColors(nStates);
      const thr = thresholds[modelSel.value];

      // readout
      $('ex-lapno').textContent = `${cur} / ${laps[laps.length - 1].lapNumber}`;
      $('ex-laptime').textContent = now?.lapTime != null ? fmtTime(now.lapTime) : '–';
      $('ex-stint').textContent = now?.stint ?? '–';
      const keptInStint = seen.filter((l) => l.stint === now?.stint && l.modelObservation).length;
      $('ex-age').textContent = keptInStint;
      const p = now?.[key];
      $('ex-belief').innerHTML = p
        ? names.map((n, i) => `<div class="belief-row"><span>${n}</span><div class="belief-bar"><div style="width:${(p[i] * 100).toFixed(1)}%;background:${colors[i]}"></div></div><span class="num">${Math.round(p[i] * 100)}%</span></div>`).join('') +
          `<div class="small muted">Alarm when P(Worn) ≥ ${thr != null ? thr : '–'}${now.modelObservation ? '' : ' · this lap was not used; belief carried over'}</div>`
        : '<div class="small muted">No belief yet in this stint (the model has not seen a usable lap since the start or the last stop).</div>';
      const tags = [];
      if (now?.actualPit) tags.push(['Real pit stop', 'var(--pit)']);
      for (const m of ORDER) if (now?.[METHODS[m].flag]) tags.push([`${METHODS[m].label} alarm`, METHODS[m].color]);
      $('ex-events').innerHTML = tags.length
        ? `This lap: ${tags.map(([t, c]) => `<span class="tag" style="background:${c};${c === 'var(--pit)' ? 'color:var(--bg)' : ''}">${t}</span>`).join('')}`
        : '<span class="muted small">No alarms or pit stops on this lap.</span>';

      drawTimes(seen, cur);
      drawProbs(seen, cur, key, colors, thr);
      drawAlarms(seen, cur);
      $('ex-legend').innerHTML = names.map((n, i) => `<span><span class="sw" style="background:${colors[i]}"></span>${n}</span>`).join('');
      summary(cur);
    }
    function fmtTime(s) {
      const m = Math.floor(s / 60);
      return m ? `${m}:${(s - m * 60).toFixed(3).padStart(6, '0')}` : `${s.toFixed(3)}s`;
    }
    function xScale(w, pad) { return linear(0.5, maxLap + 0.5, pad.l, w - pad.r); }
    const PAD = { l: 44, r: 10, t: 8, b: 22 };

    function pitLines(svg, x, H, pad, cur) {
      for (const pl of drv.actualPitLaps || []) {
        if (pl > cur) continue;
        svg.append(el('line', { x1: x(pl), x2: x(pl), y1: pad.t, y2: H - pad.b, stroke: 'var(--pit)', 'stroke-dasharray': '4 3', opacity: 0.7 }));
      }
      svg.append(el('line', { x1: x(cur), x2: x(cur), y1: pad.t, y2: H - pad.b, stroke: 'var(--accent)', 'stroke-width': 1.5, opacity: 0.6 }));
      if (cur < maxLap) svg.append(el('rect', { x: x(cur + 0.5), y: pad.t, width: Math.max(0, x(maxLap + 0.5) - x(cur + 0.5)), height: H - pad.b - pad.t, fill: 'var(--surface-2)', opacity: 0.6 }));
    }
    function lapAxis(svg, x, w, H, pad) {
      svg.append(el('line', { x1: pad.l, x2: w - pad.r, y1: H - pad.b, y2: H - pad.b, stroke: 'var(--muted)' }));
      for (const v of ticks(1, maxLap, Math.max(3, Math.floor(w / 80)))) {
        svg.append(el('text', { x: x(v), y: H - pad.b + 14, 'text-anchor': 'middle' }, String(v)));
      }
    }
    function drawTimes(seen, cur) {
      const svg = $('ex-times');
      const H = 170, w = svgSize(svg, H);
      svg.replaceChildren();
      const used = laps.filter((l) => l.modelObservation && l.lapTime != null).map((l) => l.lapTime);
      const all = laps.filter((l) => l.lapTime != null).map((l) => l.lapTime);
      const lo = Math.min(...(used.length ? used : all));
      const sorted = [...(used.length ? used : all)].sort((a, b) => a - b);
      const hi = Math.max(lo + 1, sorted[Math.floor(sorted.length * 0.98)] || lo + 1) + 0.3;
      const x = xScale(w, PAD), y = linear(lo - 0.3, hi, H - PAD.b, PAD.t);
      for (const v of ticks(lo - 0.3, hi, 3)) {
        svg.append(el('line', { x1: PAD.l, x2: w - PAD.r, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-dasharray': '2 3' }));
        svg.append(el('text', { x: PAD.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, v.toFixed(0)));
      }
      pitLines(svg, x, H, PAD, cur);
      lapAxis(svg, x, w, H, PAD);
      let path = '', prevStint = null;
      for (const l of seen) {
        if (!l.modelObservation || l.lapTime == null) continue;
        path += `${l.stint !== prevStint ? 'M' : 'L'}${x(l.lapNumber).toFixed(1)},${y(Math.min(l.lapTime, hi)).toFixed(1)}`;
        prevStint = l.stint;
      }
      svg.append(el('path', { d: path, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.2, opacity: 0.6 }));
      for (const l of seen) {
        if (l.lapTime == null) continue;
        const yy = y(Math.min(l.lapTime, hi));
        const c = el('circle', l.modelObservation
          ? { cx: x(l.lapNumber), cy: yy, r: 3, fill: 'var(--text)' }
          : { cx: x(l.lapNumber), cy: yy, r: 3, fill: 'var(--surface)', stroke: 'var(--muted)', 'stroke-width': 1.3 });
        c.append(el('title', {}, `Lap ${l.lapNumber}: ${fmtTime(l.lapTime)}${l.modelObservation ? '' : ' (not used)'}`));
        svg.append(c);
      }
    }
    function drawProbs(seen, cur, key, colors, thr) {
      const svg = $('ex-probs');
      const H = 150, w = svgSize(svg, H);
      svg.replaceChildren();
      const x = xScale(w, PAD), y = linear(0, 1, H - PAD.b, PAD.t);
      for (const v of [0, 0.5, 1]) svg.append(el('text', { x: PAD.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, `${v * 100}%`));
      const bw = Math.max(1, x(1.5) - x(0.5) - 0.5);
      for (const l of seen) {
        const p = l[key];
        if (!p) continue;
        let base = 0; // stack from the bottom: Worn first, so its height = P(Worn)
        for (let i = p.length - 1; i >= 0; i--) {
          svg.append(el('rect', { x: x(l.lapNumber) - bw / 2, y: y(base + p[i]), width: bw, height: Math.max(0, y(base) - y(base + p[i])), fill: colors[i], opacity: l.modelObservation ? 0.95 : 0.5 }));
          base += p[i];
        }
      }
      pitLines(svg, x, H, PAD, cur);
      lapAxis(svg, x, w, H, PAD);
      if (thr != null) {
        svg.append(el('line', { x1: PAD.l, x2: w - PAD.r, y1: y(thr), y2: y(thr), stroke: 'var(--text)', 'stroke-dasharray': '1 3', 'stroke-width': 2 }));
        svg.append(el('text', { x: w - PAD.r - 2, y: y(thr) - 4, 'text-anchor': 'end', style: 'fill:var(--text);font-size:10px' }, `alarm ${thr}`));
      }
    }
    function drawAlarms(seen, cur) {
      const svg = $('ex-alarms');
      const rows = [['Real pit', 'pit'], ...ORDER.map((m) => [METHODS[m].label, m])];
      const rh = 22, H = rows.length * rh + 24, w = svgSize(svg, H);
      svg.replaceChildren();
      const pad = { ...PAD, l: Math.min(150, w * 0.32) };
      const x = xScale(w, pad);
      pitLines(svg, x, H, pad, cur);
      lapAxis(svg, x, w, H, pad);
      rows.forEach(([label, m], i) => {
        const yy = pad.t + i * rh + rh / 2;
        svg.append(el('text', { x: pad.l - 6, y: yy + 4, 'text-anchor': 'end', style: 'fill:var(--text);font-size:11px' }, label));
        svg.append(el('line', { x1: pad.l, x2: w - pad.r, y1: yy, y2: yy, stroke: 'var(--line)' }));
        for (const l of seen) {
          if (m === 'Logistic' && l.logisticPositive) {
            svg.append(el('rect', { x: x(l.lapNumber) - 2, y: yy - 2, width: 4, height: 4, fill: METHODS.Logistic.color, opacity: 0.35 }));
          }
          const on = m === 'pit' ? l.actualPit : l[METHODS[m].flag];
          if (!on) continue;
          const c = m === 'pit'
            ? el('rect', { x: x(l.lapNumber) - 5, y: yy - 5, width: 10, height: 10, fill: 'var(--pit)' })
            : el('circle', { cx: x(l.lapNumber), cy: yy, r: 6, fill: METHODS[m].color });
          c.append(el('title', {}, `${label}: lap ${l.lapNumber}`));
          svg.append(c);
        }
      });
    }
    function summary(cur) {
      const last = laps[laps.length - 1].lapNumber;
      const box = $('ex-summary');
      if (cur < last) { box.innerHTML = '<p class="muted" style="margin-top:8px">Step to the last lap to see how each model scored in this race (±2 laps).</p>'; return; }
      const pits = drv.actualPitLaps || [];
      const items = ORDER.map((m) => {
        const alarms = laps.filter((l) => l[METHODS[m].flag]).map((l) => l.lapNumber);
        // greedy one-to-one matching by smallest distance (fine for a few stops per race)
        const pairs = [];
        for (const a of alarms) for (const p of pits) if (Math.abs(a - p) <= 2) pairs.push([Math.abs(a - p), a, p]);
        pairs.sort((u, v) => u[0] - v[0]);
        const ua = new Set(), up = new Set(), hit = new Map();
        for (const [, a, p] of pairs) if (!ua.has(a) && !up.has(p)) { ua.add(a); up.add(p); hit.set(a, p); }
        const parts = alarms.map((a) => (hit.has(a) ? `<span class="hit">lap ${a} ✓ (pit ${hit.get(a)})</span>` : `<span class="miss">lap ${a} ✗</span>`));
        return `<li><strong style="color:${METHODS[m].color}">${METHODS[m].label}</strong>: ${parts.length ? parts.join(', ') : '<span class="muted">no alarms</span>'} — ${up.size}/${pits.length} pits caught</li>`;
      });
      box.innerHTML = `<h3>Scorecard for this race</h3><p>Real pit stops: ${pits.length ? pits.map((p) => `lap ${p}`).join(', ') : 'none'}.</p><ul>${items.join('')}</ul>`;
    }

    function step(d) {
      const v = Math.min(+lapIn.max, Math.max(+lapIn.min, +lapIn.value + d));
      lapIn.value = v;
      render();
      return v;
    }
    function stop() { clearInterval(timer); timer = null; $('ex-play').textContent = '▶ Play'; $('ex-play').setAttribute('aria-label', 'Play'); }
    $('ex-play').addEventListener('click', () => {
      if (timer) return stop();
      if (+lapIn.value >= +lapIn.max) lapIn.value = lapIn.min;
      $('ex-play').textContent = '❚❚ Pause';
      $('ex-play').setAttribute('aria-label', 'Pause');
      timer = setInterval(() => { if (step(1) >= +lapIn.max) stop(); }, 180);
    });
    $('ex-prev').addEventListener('click', () => { stop(); step(-1); });
    $('ex-next').addEventListener('click', () => { stop(); step(1); });
    lapIn.addEventListener('input', () => { stop(); render(); });
    raceSel.addEventListener('change', () => { stop(); fillDrivers(); });
    drvSel.addEventListener('change', () => { stop(); setDriver(true); });
    modelSel.addEventListener('change', render);
    document.addEventListener('keydown', (e) => {
      if (!e.target.closest || e.target.closest('input,select,textarea')) return;
      const box = $('explorer-card').getBoundingClientRect();
      if (box.bottom < 0 || box.top > innerHeight) return;
      if (e.key === 'ArrowRight') { stop(); step(1); }
      if (e.key === 'ArrowLeft') { stop(); step(-1); }
    });
    fillDrivers();
    return render;
  }

  /* ---------- nav highlighting ----------
   * A passive scroll listener that only changes classes and scrolls the nav bar
   * sideways. It never scrolls the page, so it cannot fight the user's scrolling. */
  function initNav() {
    const ul = document.querySelector('nav.toc ul');
    const links = [...ul.querySelectorAll('a')];
    const sections = links.map((a) => document.getElementById(a.getAttribute('href').slice(1)));
    let active = null, queued = false;
    function update() {
      queued = false;
      const mark = innerHeight * 0.4;
      let idx = 0;
      sections.forEach((s, i) => { if (s && s.getBoundingClientRect().top <= mark) idx = i; });
      if (scrollY < 50) idx = -1;
      const a = links[idx] || null;
      if (a === active) return;
      if (active) active.classList.remove('active');
      active = a;
      if (!a) { ul.scrollLeft = 0; return; }
      a.classList.add('active');
      const left = a.offsetLeft - ul.offsetLeft, right = left + a.offsetWidth;
      if (left < ul.scrollLeft) ul.scrollLeft = left - 8;
      else if (right > ul.scrollLeft + ul.clientWidth) ul.scrollLeft = right - ul.clientWidth + 8;
    }
    addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }

  /* ---------- boot ---------- */
  async function main() {
    initNav();
    const redraw = [];
    if (location.protocol === 'file:') {
      showError('<strong>Open this page through a web server.</strong> Browsers block data files when a page is opened from disk. Run <code>python -m http.server -d docs 8000</code> and visit <code>http://localhost:8000</code>.');
    }
    const safe = async (label, fn) => {
      try { await fn(); } catch (err) {
        console.error(err);
        showError(`<strong>Could not load ${esc(label)}.</strong> ${esc(err.message)}`);
      }
    };
    const placeholderFlag = await getJSON('placeholder.json', true).catch(() => null);
    if (placeholderFlag?.placeholder) markPlaceholder();

    let val = null, pooled = null;
    await safe('2019 validation metrics', async () => {
      val = await getJSON('validation-threshold-metrics-2019.json');
      if (val.placeholder) markPlaceholder();
      redraw.push(initThreshold(val));
    });
    await safe('2020 results', async () => {
      pooled = await getCSV('evaluation_pooled_summary.csv');
      redraw.push(initResults(pooled));
    });
    await safe('timing deltas', async () => {
      redraw.push(initTiming(await getCSV('prediction_timing_deltas.csv')));
    });
    await safe('race explorer data', async () => {
      $('ex-belief').innerHTML = '<p class="loading">Loading race data (a few MB)…</p>';
      const [explorer, drivers] = await Promise.all([
        getJSON('race-explorer-2020.json'),
        getJSON('drivers.json', true).then((d) => d || {}),
      ]);
      if (explorer.placeholder) markPlaceholder();
      // Pipeline writes driver ids as numbers; the <select> works with strings.
      explorer.races.forEach((r) => r.drivers.forEach((d) => { d.driver = String(d.driver); }));
      redraw.push(initExplorer(explorer, drivers, val));
      if (pooled) finalStintTable(explorer, pooled);
    });

    let t;
    let lastW = innerWidth;
    addEventListener('resize', () => {
      if (innerWidth === lastW) return; // ignore mobile URL-bar height changes
      lastW = innerWidth;
      clearTimeout(t);
      t = setTimeout(() => redraw.forEach((f) => f()), 150);
    });
  }
  main();
})();
