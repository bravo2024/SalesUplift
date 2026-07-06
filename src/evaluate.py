from __future__ import annotations
"""Evaluation + reporting for uplift metalearners."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from src.core import (
    qini_coefficient, auuc, uplift_at_k, expected_response_lift,
    average_treatment_effect, qini_curve, random_qini_curve,
)


def uplift_report(fit_result: dict, ks=(0.1, 0.2, 0.3)) -> dict:
    """Compute per-model uplift metrics on the held-out test fold."""
    yte = fit_result["y_test"]
    wte = fit_result["treatment_test"]
    ttau = fit_result["true_tau_test"]
    per_model = {}
    for name, up in fit_result["uplift_test"].items():
        rho = spearmanr(up, ttau).correlation if len(ttau) > 2 else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        per_model[name] = {
            "qini_coefficient": qini_coefficient(yte, wte, up),
            "auuc": auuc(yte, wte, up),
            **{f"uplift_at_{int(kk * 100)}pct": uplift_at_k(yte, wte, up, k=kk) for kk in ks},
            "expected_response_lift_30pct": expected_response_lift(yte, wte, up, k=0.3),
            "spearman_vs_true_tau": float(rho),
            "cate_mean": float(np.mean(up)),
            "cate_std": float(np.std(up)),
        }
    return {"per_model": per_model, "ate_observed": average_treatment_effect(yte, wte)}


def qini_curves_for_plot(fit_result: dict, n_points: int = 100) -> dict:
    """Qini curves (model + random baseline) for visualization."""
    yte, wte = fit_result["y_test"], fit_result["treatment_test"]
    out = {}
    _, base = random_qini_curve(yte, wte, n_points=n_points)
    out["random"] = base
    for name, up in fit_result["uplift_test"].items():
        _, q = qini_curve(yte, wte, up, n_points=n_points)
        out[name] = q
    return out


def _to_native(o):
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save_metrics(metrics, path: str = "models/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_native(metrics), f, indent=2)


def print_report(report: dict) -> None:
    for name, m in report["per_model"].items():
        print(f"\n{'=' * 52}\n  {name}\n{'=' * 52}")
        for k, v in m.items():
            print(f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k:28s}: {v}")
    print(f"\n  observed ATE (test)        : {report['ate_observed']:.4f}")
