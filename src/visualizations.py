from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

THEME = {
    "bg": "#0e1117",
    "fg": "#ffffff",
    "grid": "#1a1f2e",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "orange": "#f97316",
    "rose": "#f43f5e",
    "amber": "#fbbf24",
    "green": "#22c55e",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["fg"],
        "text.color": THEME["fg"],
        "xtick.color": THEME["fg"],
        "ytick.color": THEME["fg"],
        "grid.color": THEME["grid"],
        "grid.alpha": 0.3,
        "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": THEME["grid"],
        "legend.labelcolor": THEME["fg"],
    })


def plot_roc_curve(y_true, y_proba_dict):
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"], THEME["rose"]]
    for (name, y_proba), c in zip(y_proba_dict.items(), colors):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#555", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 — Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves", color=THEME["fg"])
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    return fig


def plot_precision_recall_curve(y_true, y_proba_dict):
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"], THEME["rose"]]
    baseline = y_true.mean()
    for (name, y_proba), c in zip(y_proba_dict.items(), colors):
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(rec, prec, color=c, lw=2, label=f"{name} (baseline={baseline:.3f})")
    ax.axhline(baseline, color="#555", ls=":", lw=1.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", color=THEME["fg"])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    _style()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    c = LinearSegmentedColormap.from_list("cm", [THEME["bg"], THEME["cyan"]], N=256)
    im = ax.imshow(cm, cmap=c, vmin=0, vmax=cm.max() * 1.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Good", "Predicted Default"])
    ax.set_yticklabels(["Actual Good", "Actual Default"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=THEME["fg"], fontsize=14, fontweight="bold")
    ax.set_title(title, color=THEME["fg"])
    plt.colorbar(im, ax=ax, shrink=0.8)
    return fig


def plot_score_distribution(y_proba, y_true, threshold=0.5):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    for label, mask, color, pos in [(0, y_true == 0, THEME["green"], "Good"),
                                     (1, y_true == 1, THEME["rose"], "Default")]:
        ax.hist(y_proba[mask], bins=bins, alpha=0.6, color=color,
                label=f"{pos} (n={mask.sum()})", density=True)
    ax.axvline(threshold, color=THEME["amber"], ls="--", lw=2, label=f"Threshold={threshold}")
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution", color=THEME["fg"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def plot_ks_curve(y_true, y_proba):
    _style()
    ix = np.argsort(y_proba)
    y_sorted = y_true[ix]
    n_total = len(y_sorted)
    cum_event = np.cumsum(y_sorted) / y_sorted.sum()
    cum_non_event = np.cumsum(1 - y_sorted) / (n_total - y_sorted.sum())
    ks = np.max(np.abs(cum_event - cum_non_event))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_total) / n_total * 100
    ax.plot(x, cum_event, color=THEME["rose"], lw=2, label="Defaults")
    ax.plot(x, cum_non_event, color=THEME["green"], lw=2, label="Non-defaults")
    ks_idx = np.argmax(np.abs(cum_event - cum_non_event))
    ax.axvline(x[ks_idx], color=THEME["amber"], ls=":", alpha=0.7)
    ax.annotate(f"KS={ks:.3f}", xy=(x[ks_idx], (cum_event[ks_idx] + cum_non_event[ks_idx]) / 2),
                fontsize=10, color=THEME["amber"], ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1f2e", edgecolor=THEME["amber"]))
    ax.set_xlabel("Population Percentile")
    ax.set_ylabel("Cumulative %")
    ax.set_title("Kolmogorov-Smirnov (KS) Curve", color=THEME["fg"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def plot_feature_importance(importances, features, title="Feature Importance", color=None):
    _style()
    imp = np.array([v["mean"] if isinstance(v, dict) else v for v in importances])
    idx = np.argsort(np.abs(imp))
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.3)))
    bars = ax.barh(range(len(features)), imp[idx], color=color or THEME["cyan"], alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title, color=THEME["fg"])
    ax.grid(True, alpha=0.2, axis="x")
    return fig


def plot_calibration_curve(y_true, y_proba_dict, n_bins=10):
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"], THEME["rose"]]
    for (name, y_proba), c in zip(y_proba_dict.items(), colors):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins[1:-1])
        mean_pred = np.array([y_proba[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else bins[i]
                              for i in range(n_bins)])
        frac_pos = np.array([y_true[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else 0
                             for i in range(n_bins)])
        ax.plot(mean_pred, frac_pos, "o-", color=c, lw=2, ms=4, label=name)
    ax.plot([0, 1], [0, 1], ":", color="#555", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration (Reliability) Curve", color=THEME["fg"])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
    return fig


def plot_portfolio_risk(segments):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(segments.keys())
    counts = [seg["count"] for seg in segments.values()]
    rates = [seg["default_rate"] for seg in segments.values()]
    axes[0].bar(names, counts, color=THEME["cyan"], alpha=0.7, edgecolor=THEME["grid"])
    axes[0].set_title("Loans per Segment", color=THEME["fg"])
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].grid(True, alpha=0.2, axis="y")
    colors_bar = [THEME["green"] if r < 0.15 else THEME["amber"] if r < 0.3 else THEME["rose"] for r in rates]
    axes[1].bar(names, rates, color=colors_bar, alpha=0.7, edgecolor=THEME["grid"])
    axes[1].axhline(y=sum(rates) / len(rates), color=THEME["fg"], ls="--", lw=1, alpha=0.5)
    axes[1].set_title("Default Rate per Segment", color=THEME["fg"])
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    return fig
