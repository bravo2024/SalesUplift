from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import xgboost as xgb


def compute_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["gini"] = 2.0 * metrics["roc_auc"] - 1.0
    return metrics


def ks_statistic(y_true, y_score):
    ix = np.argsort(y_score)
    y_true_sorted = y_true[ix]
    y_score_sorted = y_score[ix]
    n_total = len(y_true_sorted)
    n_event = y_true_sorted.sum()
    n_non_event = n_total - n_event
    cum_event = np.cumsum(y_true_sorted) / n_event
    cum_non_event = np.cumsum(1 - y_true_sorted) / n_non_event
    return np.max(np.abs(cum_event - cum_non_event))


def population_stability_index(expected, actual, n_bins=10):
    eps = 1e-10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = np.digitize(np.clip(np.concatenate([expected, actual]), 0, 0.999), bins[1:-1]) - 1
    expected_counts = np.bincount(bin_labels[:len(expected)], minlength=n_bins)
    actual_counts = np.bincount(bin_labels[len(expected):], minlength=n_bins)
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    psi = np.sum((actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps)))
    return psi


def build_models(X_train, y_train, seed=42):
    lr = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", random_state=seed, max_iter=1000)
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    gbt = GradientBoostingClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                     learning_rate=0.05, subsample=0.8, random_state=seed)
    gbt.fit(X_train, y_train)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train.mean() > 0.1) * 1.0 + (y_train.mean() <= 0.1) * 3.0,
        eval_metric="logloss", random_state=seed,
    )
    xgb_model.fit(X_train, y_train)
    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gbt,
        "XGBoost": xgb_model,
    }


def woe_transform(df, feature, target, min_samples=5):
    if df[feature].dtype == object or df[feature].nunique() < 10:
        groups = df.groupby(feature)[target]
    else:
        df["_bin"] = pd.qcut(df[feature], 10, duplicates="drop")
        groups = df.groupby("_bin")[target]
    result = groups.agg(["count", "sum"])
    result.columns = ["count", "event"]
    result = result[result["count"] >= min_samples]
    result["non_event"] = result["count"] - result["event"]
    n_event_total = result["event"].sum()
    n_non_event_total = result["non_event"].sum()
    result["event_rate"] = (result["event"] + 0.5) / (n_event_total + 0.5)
    result["non_event_rate"] = (result["non_event"] + 0.5) / (n_non_event_total + 0.5)
    result["woe"] = np.log(result["event_rate"] / result["non_event_rate"])
    result["iv"] = (result["event_rate"] - result["non_event_rate"]) * result["woe"]
    return result
