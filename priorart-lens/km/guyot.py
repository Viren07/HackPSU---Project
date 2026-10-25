"""Reconstruct individual patient data (IPD) from a digitized Kaplan-Meier curve.

Implements the algorithm of Guyot, Ades, Ouwens & Welton (2012),
"Enhanced secondary analysis of survival data: reconstructing the data from
published Kaplan-Meier survival curves", BMC Medical Research Methodology 12:9.

Inputs
  t_s, s      digitized curve: times and survival probabilities (0..1)
  t_risk      times where the number-at-risk table reports counts
  n_risk      the counts
  tot_events  optional total number of events (improves the last interval)

Idea
  Between two at-risk times, the drop in the number at risk = events + censorings.
  The curve tells us how big each event step is; the at-risk table tells us how
  many people left in total. So we guess the censorings, spread them evenly over
  the interval, rebuild the curve, and adjust the guess until the rebuilt
  number at risk matches the published one.
"""
from __future__ import annotations

import numpy as np


def clean_curve(t, s):
    """Sort, force S(0)=1, make the curve non-increasing, drop duplicates."""
    t, s = np.asarray(t, float), np.clip(np.asarray(s, float), 0, 1)
    o = np.argsort(t, kind="stable")
    t, s = t[o], s[o]
    keep = t >= 0
    t, s = t[keep], s[keep]
    if len(t) == 0 or t[0] > 0:
        t, s = np.r_[0.0, t], np.r_[1.0, s]
    else:
        s[0] = 1.0
    s = np.minimum.accumulate(s)
    # remove exact duplicate (t, s) pairs
    m = np.r_[True, (np.diff(t) > 0) | (np.diff(s) != 0)]
    return t[m], s[m]


def _interval_bounds(t_s, t_risk):
    nint = len(t_risk)
    lower = np.array([int(np.searchsorted(t_s, tr, side="left")) for tr in t_risk])
    lower = np.minimum(lower, len(t_s) - 1)
    upper = np.r_[lower[1:] - 1, len(t_s) - 1]
    upper = np.maximum(upper, lower)
    return lower, upper, nint


def _run_interval(i, lower, upper, t_s, s, n_censor_i, n_hat, d, cen, km_hat, last, t_next):
    """Place censorings evenly and compute events for interval i."""
    lo, up = lower[i], upper[i]
    cen[lo:up + 1] = 0
    if n_censor_i > 0:
        t_end = t_next if t_next is not None else t_s[up]
        cen_t = t_s[lo] + (np.arange(1, n_censor_i + 1)) * (t_end - t_s[lo]) / (n_censor_i + 1)
        for k in range(lo, up + 1):
            right = t_s[k + 1] if k + 1 < len(t_s) else np.inf
            cen[k] = int(((cen_t >= t_s[k]) & (cen_t < right)).sum())
    for k in range(lo, up + 1):
        if k == 0:
            d[k] = 0
            km_hat[k] = 1.0
        else:
            nk = max(n_hat[k], 0)
            prev = km_hat[last] if km_hat[last] > 0 else 1e-9
            d[k] = int(round(nk * (1 - s[k] / prev))) if nk > 0 else 0
            d[k] = int(np.clip(d[k], 0, nk))
            km_hat[k] = km_hat[last] * (1 - d[k] / nk) if nk > 0 else km_hat[last]
        n_hat[k + 1] = n_hat[k] - d[k] - cen[k]
        if d[k] != 0:
            last = k
    return last


def reconstruct(t_s, s, t_risk, n_risk, tot_events=None, max_iter=100):
    """Return (times, events) arrays: one row per reconstructed patient."""
    t_s, s = clean_curve(t_s, s)
    t_risk = np.asarray(t_risk, float)
    n_risk = np.asarray(n_risk, int)
    # keep only at-risk points that fall inside the digitized follow-up
    ok = t_risk <= t_s[-1] + 1e-9
    t_risk, n_risk = t_risk[ok], n_risk[ok]
    # drop trailing zeros (nobody left), keeping one
    while len(n_risk) > 1 and n_risk[-1] == 0 and n_risk[-2] == 0:
        t_risk, n_risk = t_risk[:-1], n_risk[:-1]

    lower, upper, nint = _interval_bounds(t_s, t_risk)
    m = len(t_s)
    n_hat = np.zeros(m + 1)
    d = np.zeros(m, int)
    cen = np.zeros(m, int)
    km_hat = np.ones(m)
    n_censor = np.zeros(nint, int)
    last = 0
    n_hat[0] = n_risk[0]

    for i in range(nint - 1):
        last_start = last
        n_hat[lower[i]] = n_risk[i]
        s_lo = s[lower[i]] if s[lower[i]] > 0 else 1e-9
        n_censor[i] = max(0, int(round(n_risk[i] * s[lower[i + 1]] / s_lo - n_risk[i + 1])))
        for _ in range(max_iter):
            last = last_start
            last = _run_interval(i, lower, upper, t_s, s, n_censor[i], n_hat, d, cen, km_hat, last, t_s[lower[i + 1]])
            diff = int(round(n_hat[lower[i + 1]] - n_risk[i + 1]))
            if diff == 0:
                break
            new = max(0, n_censor[i] + diff)
            if new == n_censor[i]:
                break
            n_censor[i] = new

    # last interval: no count at its end
    i = nint - 1
    n_hat[lower[i]] = n_risk[i]
    if tot_events is not None:
        ev_so_far = int(d[:lower[i]].sum())
        # choose censorings so total events match
        best, best_err = 0, None
        for nc in range(0, int(n_risk[i]) + 1):
            l2 = _run_interval(i, lower, upper, t_s, s, nc, n_hat, d, cen, km_hat, last, None)
            err = abs(ev_so_far + int(d[lower[i]:upper[i] + 1].sum()) - tot_events)
            if best_err is None or err < best_err:
                best, best_err = nc, err
            if err == 0:
                break
        n_censor[i] = best
    else:
        # assume the censoring rate seen so far continues
        span_prev = t_s[lower[i]] - t_s[0]
        rate = n_censor[:i].sum() / span_prev if span_prev > 0 else 0
        n_censor[i] = int(min(round(rate * (t_s[upper[i]] - t_s[lower[i]])), n_risk[i]))
    last = _run_interval(i, lower, upper, t_s, s, n_censor[i], n_hat, d, cen, km_hat, last, None)

    # anyone still at risk at the end is censored at the last time
    remaining = int(max(0, round(n_hat[upper[i] + 1])))
    times, events = [], []
    for k in range(m):
        times += [t_s[k]] * int(d[k]) + []
        events += [1] * int(d[k])
    # censoring times: evenly spread within each step, as in the algorithm
    for k in range(m):
        if cen[k] > 0:
            right = t_s[k + 1] if k + 1 < m else t_s[k]
            ct = t_s[k] + (np.arange(1, cen[k] + 1)) * (right - t_s[k]) / (cen[k] + 1)
            times += ct.tolist()
            events += [0] * int(cen[k])
    if remaining > 0:
        times += [t_s[-1]] * remaining
        events += [0] * remaining
    times, events = np.array(times, float), np.array(events, int)
    # rounding inside the intervals can leave the total a patient or two off the published
    # starting count; fix it by adjusting censored patients at the end of follow-up, never deaths
    n0 = int(n_risk[0])
    diff = len(times) - n0
    if diff > 0:
        cens_idx = np.where(events == 0)[0]
        drop = cens_idx[np.argsort(-times[cens_idx])][:diff]
        keep = np.ones(len(times), bool)
        keep[drop] = False
        times, events = times[keep], events[keep]
    elif diff < 0:
        times = np.r_[times, [t_s[-1]] * (-diff)]
        events = np.r_[events, [0] * (-diff)]
    return times, events
