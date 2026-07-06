from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr

from src.data import make_synthetic
from src.model import fit_uplift_models, predict_uplift
from src.core import (
    qini_coefficient, auuc, uplift_at_k, expected_response_lift,
    qini_curve, average_treatment_effect,
)


def test_data_has_treatment_and_heterogeneous_tau():
    d = make_synthetic(800, seed=0)
    assert "treatment" in d["df"].columns
    assert 0.40 < d["treatment_rate"] < 0.60          # randomized ~50/50
    assert d["true_tau"].std() > 1e-3                  # genuine CATE heterogeneity
    assert 0.0 < d["positive_rate"] < 1.0
    assert -0.1 < d["ate"] < 0.6


def test_learners_predict_cate_of_correct_shape():
    d = make_synthetic(900, seed=1)
    fit = fit_uplift_models(d, seed=1)
    assert set(fit["uplift_test"]) == {"T-learner", "S-learner"}
    for name, up in fit["uplift_test"].items():
        assert up.shape[0] == fit["n_test"]
        assert np.all(np.isfinite(up))
    # predict_uplift works on raw rows too
    up = predict_uplift(fit["models"]["T-learner"], d["X"].iloc[:5])
    assert up.shape == (5,)


def test_uplift_metrics_are_finite():
    d = make_synthetic(900, seed=2)
    fit = fit_uplift_models(d, seed=2)
    yte, wte = fit["y_test"], fit["treatment_test"]
    up = fit["uplift_test"]["T-learner"]
    qc = qini_coefficient(yte, wte, up)
    au = auuc(yte, wte, up)
    uk = uplift_at_k(yte, wte, up, k=0.3)
    er = expected_response_lift(yte, wte, up, k=0.3)
    assert np.isfinite(qc) and np.isfinite(au) and np.isfinite(er)
    assert -1.0 <= uk <= 1.0


def test_qini_curve_origin_and_shape():
    d = make_synthetic(700, seed=4)
    fit = fit_uplift_models(d, seed=4)
    f, q = qini_curve(fit["y_test"], fit["treatment_test"], fit["uplift_test"]["S-learner"])
    assert f.shape == q.shape
    assert f[0] == 0.0 and q[0] == 0.0
    assert abs(f[-1] - 1.0) < 1e-9


def test_predicted_uplift_correlates_with_true_cate():
    d = make_synthetic(3000, seed=3)
    fit = fit_uplift_models(d, seed=3)
    best = -1.0
    for name in ("T-learner", "S-learner"):
        rho = spearmanr(fit["uplift_test"][name], fit["true_tau_test"]).correlation
        best = max(best, 0.0 if not np.isfinite(rho) else rho)
    assert best > 0.0, "at least one metalearner must recover CATE signal"


def test_ate_estimator_runs():
    d = make_synthetic(500, seed=5)
    ate = average_treatment_effect(d["y"], d["treatment"])
    assert np.isfinite(ate)
