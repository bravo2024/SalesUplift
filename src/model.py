from __future__ import annotations
"""Uplift metalearners for heterogeneous treatment-effect estimation.

Implements the two workhorse metalearners of Künzel et al. (2019):
  * T-learner: fit mu1 on treated units and mu0 on control units separately,
    then estimate CATE(x) = mu1(x) - mu0(x).
  * S-learner: fit a single response model on [X, W], then estimate
    CATE(x) = p(y=1 | x, W=1) - p(y=1 | x, W=0).

Both use gradient-boosted trees (LightGBM when available, falling back to
scikit-learn's GradientBoostingClassifier) as the base response estimator.
Categorical covariates are one-hot encoded with a fitted encoder so train/test
column alignment is guaranteed.
"""
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover - fallback path
    _HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def _gbm(seed: int):
    """Regularized gradient-boosted classifier (LightGBB if present else sklearn)."""
    if _HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1,
        )
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        min_samples_leaf=20, random_state=seed,
    )


class FeatureEncoder:
    """One-hot encode a fixed set of categorical columns; pass others through."""

    def __init__(self, categorical_features):
        self.cat = list(categorical_features)
        self.columns_: list[str] = []

    def _frame(self, X) -> pd.DataFrame:
        Xc = X.copy()
        for c in self.cat:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype(str)
        if self.cat:
            return pd.get_dummies(Xc, columns=self.cat, drop_first=False)
        return Xc

    def fit(self, X):
        self.columns_ = self._frame(X).columns.tolist()
        return self

    def transform(self, X) -> pd.DataFrame:
        Xt = self._frame(X)
        for c in self.columns_:
            if c not in Xt.columns:
                Xt[c] = 0
        return Xt[self.columns_].astype(float)


class TLearner:
    """Two-model metalearner: CATE(x) = mu1(x) - mu0(x)."""

    name = "T-learner"

    def __init__(self, categorical_features, seed: int = 42):
        self.cat = list(categorical_features)
        self.encoder = FeatureEncoder(self.cat)
        self.mu1 = _gbm(seed)
        self.mu0 = _gbm(seed + 1)

    def fit(self, X, y, treatment):
        self.encoder.fit(X)
        t = np.asarray(treatment)
        ya = np.asarray(y, dtype=float)
        Xe = self.encoder.transform(X)
        self.mu1.fit(Xe[t == 1], ya[t == 1])
        self.mu0.fit(Xe[t == 0], ya[t == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        Xe = self.encoder.transform(X)
        return self.mu1.predict_proba(Xe)[:, 1] - self.mu0.predict_proba(Xe)[:, 1]


class SLearner:
    """Single-model metalearner: CATE(x) = p(x,W=1) - p(x,W=0)."""

    name = "S-learner"
    _tcol = "__treatment__"

    def __init__(self, categorical_features, seed: int = 42):
        self.cat = list(categorical_features)
        self.encoder = FeatureEncoder(self.cat)
        self.model = _gbm(seed)

    @staticmethod
    def _augment(X, treatment) -> pd.DataFrame:
        Xa = X.copy()
        Xa[SLearner._tcol] = np.asarray(treatment, dtype=float)
        return Xa

    def fit(self, X, y, treatment):
        Xa = self._augment(X, treatment)
        self.encoder.fit(Xa)
        Xe = self.encoder.transform(Xa)
        self.model.fit(Xe, np.asarray(y, dtype=float))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        n = len(X)
        p1 = self.model.predict_proba(self.encoder.transform(self._augment(X, np.ones(n))))[:, 1]
        p0 = self.model.predict_proba(self.encoder.transform(self._augment(X, np.zeros(n))))[:, 1]
        return p1 - p0


def fit_uplift_models(data: dict, seed: int = 42, test_size: float = 0.25) -> dict:
    """Fit T- and S-learners on a stratified train/test split (stratify by treatment
    to keep the ~50/50 assignment in both folds)."""
    X = data["X"]
    y = np.asarray(data["y"], dtype=float)
    w = np.asarray(data["treatment"], dtype=float)
    cat = data.get("categorical_features", [])
    true_tau = data.get("true_tau")
    if true_tau is not None:
        true_tau = np.asarray(true_tau, dtype=float)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=w, random_state=seed)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte, wtr, wte = y[tr], y[te], w[tr], w[te]

    t_learner = TLearner(cat, seed=seed).fit(Xtr, ytr, wtr)
    s_learner = SLearner(cat, seed=seed).fit(Xtr, ytr, wtr)

    models = {"T-learner": t_learner, "S-learner": s_learner}
    uplift_test = {name: predict_uplift(m, Xte) for name, m in models.items()}
    uplift_train = {name: predict_uplift(m, Xtr) for name, m in models.items()}

    return {
        "models": models,
        "X_train": Xtr, "X_test": Xte,
        "y_train": ytr, "y_test": yte,
        "treatment_train": wtr, "treatment_test": wte,
        "true_tau_test": true_tau[te] if true_tau is not None else None,
        "uplift_test": uplift_test,
        "uplift_train": uplift_train,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "categorical_features": list(cat),
        "features": list(X.columns),
    }


def predict_uplift(model, X) -> np.ndarray:
    """Predict CATE (uplift) for arbitrary feature rows."""
    return np.asarray(model.predict_uplift(X), dtype=float)
