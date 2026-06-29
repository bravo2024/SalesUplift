from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from src.core import build_models, compute_metrics, ks_statistic


def train_all_models(data, seed=42, test_size=0.25):
    X = data["X"].copy()
    y = data["y"].values if hasattr(data["y"], "values") else data["y"].copy()
    cat_cols = data.get("categorical_features", [])
    for c in cat_cols:
        if c in X.columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
    num_cols = data.get("numerical_features", [])
    for c in num_cols:
        if c in X.columns:
            X[c] = X[c].fillna(X[c].median())
    from sklearn.model_selection import train_test_split as _tts
    X_train, X_test, y_train, y_test = _tts(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    num_cols_actual = [c for c in num_cols if c in X_train.columns]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    if num_cols_actual:
        X_train_scaled[num_cols_actual] = scaler.fit_transform(X_train[num_cols_actual])
        X_test_scaled[num_cols_actual] = scaler.transform(X_test[num_cols_actual])
    models = build_models(X_train_scaled, y_train, seed=seed)
    results = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["ks"] = ks_statistic(y_test, y_proba)
        results[name] = {"metrics": metrics, "y_proba": y_proba, "y_pred": y_pred}
    return {
        "models": models,
        "results": results,
        "scaler": scaler,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "features": list(X.columns),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def cross_validate(data, seed=42, n_folds=5):
    X = data["X"].copy()
    y = data["y"].values if hasattr(data["y"], "values") else data["y"].copy()
    cat_cols = data.get("categorical_features", [])
    for c in cat_cols:
        if c in X.columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
    num_cols = data.get("numerical_features", [])
    for c in num_cols:
        if c in X.columns:
            X[c] = X[c].fillna(X[c].median())
    num_cols_actual = [c for c in num_cols if c in X.columns]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = {name: {"roc_auc": [], "gini": [], "ks": [], "f1": []}
                  for name in ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]}
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        if num_cols_actual:
            X_tr_scaled = X_tr.copy()
            X_te_scaled = X_te.copy()
            X_tr_scaled[num_cols_actual] = scaler.fit_transform(X_tr[num_cols_actual])
            X_te_scaled[num_cols_actual] = scaler.transform(X_te[num_cols_actual])
        else:
            X_tr_scaled, X_te_scaled = X_tr, X_te
        models = build_models(X_tr_scaled, y_tr, seed=seed)
        for name, model in models.items():
            y_proba = model.predict_proba(X_te_scaled)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            met = compute_metrics(y_te, y_pred, y_proba)
            cv_results[name]["roc_auc"].append(met.get("roc_auc", 0))
            cv_results[name]["gini"].append(met.get("gini", 0))
            cv_results[name]["ks"].append(ks_statistic(y_te, y_proba))
            cv_results[name]["f1"].append(met.get("f1", 0))
    summary = {}
    for name, scores in cv_results.items():
        summary[name] = {}
        for metric, vals in scores.items():
            summary[name][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }
    return summary


def permutation_importance(model, X_val, y_val, metric_fn, n_repeats=10, seed=42):
    rng = np.random.default_rng(seed)
    baseline = metric_fn(y_val, model.predict_proba(X_val)[:, 1])
    importances = []
    for col_idx in range(X_val.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
            score = metric_fn(y_val, model.predict_proba(X_perm)[:, 1])
            scores.append(baseline - score)
        importances.append({
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        })
    return importances


def threshold_sweep(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for tau in thresholds:
        y_pred = (y_proba >= tau).astype(int)
        met = compute_metrics(y_true, y_pred, y_proba)
        met["threshold"] = float(tau)
        met["accept_rate"] = float((y_pred == 0).mean())
        rows.append(met)
    return pd.DataFrame(rows)
