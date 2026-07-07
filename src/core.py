from __future__ import annotations
"""Uplift / causal evaluation metrics, implemented from scratch.

Notation follows the uplift literature (Radcliffe & Surry 2011; Gutierrez &
Gérardy 2017). Units are sorted by predicted uplift (descending) and, as we
target an increasing top fraction f of the population, we track cumulative
treated/control outcomes to build:
  * the Qini curve  q(f) = Y_t(f) - Y_c(f) * N_t(f)/N_c(f)   (incremental
    responses captured among the targeted top-f fraction),
  * the uplift curve = treated conversion rate - control conversion rate among
    the targeted top-f fraction.
The Qini coefficient is the area between the model Qini curve and the random
(straight-line) baseline; AUUC is the area under the uplift curve.
"""
import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0; stay compatible with both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _sorted(y, treatment, uplift):
    y = np.asarray(y, dtype=float)
    w = np.asarray(treatment, dtype=float)
    u = np.asarray(uplift, dtype=float)
    order = np.argsort(-u, kind="mergesort")  # descending by predicted uplift
    return y[order], w[order], u[order]


def _subsample(frac, vals, n_points):
    if n_points and len(frac) > n_points:
        idx = np.linspace(0, len(frac) - 1, n_points).astype(int)
        return frac[idx], vals[idx]
    return frac, vals


def qini_total(y, treatment) -> float:
    """Total incremental responses if the whole population were treated."""
    y, w = np.asarray(y, float), np.asarray(treatment, float)
    nt, nc = w.sum(), (1 - w).sum()
    ratio = nt / nc if nc > 0 else 0.0
    return float((y * w).sum() - (y * (1 - w)).sum() * ratio)


def qini_curve(y, treatment, uplift, n_points: int = 100):
    """Cumulative incremental-response Qini curve vs. fraction targeted.

    Returns (fractions, qini_values) with the (0, 0) origin prepended.
    """
    y, w, _ = _sorted(y, treatment, uplift)
    n = len(y)
    cum_t = np.cumsum(w)
    cum_c = np.cumsum(1 - w)
    cum_yt = np.cumsum(y * w)
    cum_yc = np.cumsum(y * (1 - w))
    ratio = np.divide(cum_t, cum_c, out=np.zeros_like(cum_t), where=cum_c > 0)
    qini = cum_yt - cum_yc * ratio
    frac = np.arange(1, n + 1) / n
    frac, qini = _subsample(frac, qini, n_points)
    frac = np.concatenate([[0.0], frac])
    qini = np.concatenate([[0.0], qini])
    return frac, qini


def random_qini_curve(y, treatment, n_points: int = 100):
    """Straight-line baseline Qini: linear from (0,0) to (1, qini_total)."""
    total = qini_total(y, treatment)
    # n_points + 1 so it lines up with qini_curve, which prepends the origin
    frac = np.linspace(0, 1, n_points + 1)
    return frac, frac * total


def qini_coefficient(y, treatment, uplift, n_points: int = 100) -> float:
    """Area between the model Qini curve and the random baseline (>0 is skilful)."""
    frac, qini = qini_curve(y, treatment, uplift, n_points)
    _, baseline = random_qini_curve(y, treatment, n_points=len(frac))
    # resample baseline to the model's fraction grid
    baseline = np.interp(frac, np.linspace(0, 1, len(baseline)), baseline)
    return float(_trapezoid(qini - baseline, frac))


def uplift_curve(y, treatment, uplift, n_points: int = 100):
    """Treated-minus-control conversion rate among the targeted top-f fraction."""
    y, w, _ = _sorted(y, treatment, uplift)
    n = len(y)
    cum_t = np.cumsum(w)
    cum_c = np.cumsum(1 - w)
    cum_yt = np.cumsum(y * w)
    cum_yc = np.cumsum(y * (1 - w))
    rate_t = np.divide(cum_yt, cum_t, out=np.zeros_like(cum_yt), where=cum_t > 0)
    rate_c = np.divide(cum_yc, cum_c, out=np.zeros_like(cum_yc), where=cum_c > 0)
    lift = rate_t - rate_c
    frac = np.arange(1, n + 1) / n
    frac, lift = _subsample(frac, lift, n_points)
    frac = np.concatenate([[0.0], frac])
    lift = np.concatenate([[0.0], lift])
    return frac, lift


def auuc(y, treatment, uplift, n_points: int = 100) -> float:
    """Area Under the Uplift Curve (average targeted uplift across fractions)."""
    frac, lift = uplift_curve(y, treatment, uplift, n_points)
    return float(_trapezoid(lift, frac))


def uplift_at_k(y, treatment, uplift, k: float = 0.3) -> float:
    """Conversion-rate gap (treated - control) among the top-k% by predicted uplift."""
    y, w, _ = _sorted(y, treatment, uplift)
    m = max(1, int(round(k * len(y))))
    yt = (y[:m] * w[:m]).sum()
    yc = (y[:m] * (1 - w[:m])).sum()
    nt = w[:m].sum()
    nc = (1 - w[:m]).sum()
    rt = yt / nt if nt > 0 else 0.0
    rc = yc / nc if nc > 0 else 0.0
    return float(rt - rc)


def expected_response_lift(y, treatment, uplift, k: float = 0.3) -> float:
    """Expected number of incremental conversions from treating the top-k fraction."""
    n = len(y)
    return float(uplift_at_k(y, treatment, uplift, k) * (k * n))


def average_treatment_effect(y, treatment) -> float:
    """Naive difference-in-means ATE (unbiased here because treatment is randomized)."""
    y = np.asarray(y, float)
    w = np.asarray(treatment, float)
    if w.sum() == 0 or (1 - w).sum() == 0:
        return 0.0
    return float(y[w == 1].mean() - y[w == 0].mean())
