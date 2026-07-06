from __future__ import annotations
"""SalesUplift — causal uplift modeling dashboard.

Quantifies the heterogeneous treatment effect of a marketing campaign using
T-learner / S-learner metalearners and evaluates them with Qini / AUUC curves.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import make_synthetic
from src.model import fit_uplift_models
from src.evaluate import uplift_report, qini_curves_for_plot
from src.core import uplift_at_k

st.set_page_config(page_title="SalesUplift | Causal Uplift Platform", layout="wide", page_icon="📈")

BG, PANEL, FG, GRID = "#0f172a", "#1e293b", "#e2e8f0", "#334155"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color": GRID, "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})
COLORS = {"T-learner": "#22d3ee", "S-learner": "#a78bfa", "random": "#64748b"}


@st.cache_data(show_spinner="Generating synthetic campaign data…")
def get_data(n: int, seed: int):
    return make_synthetic(n=n, seed=seed)


@st.cache_data(show_spinner="Training uplift metalearners…")
def get_fit(n: int, seed: int, test_size: float):
    data = make_synthetic(n=n, seed=seed)
    fit = fit_uplift_models(data, seed=seed, test_size=test_size)
    report = uplift_report(fit)
    curves = qini_curves_for_plot(fit)
    return fit, report, curves, data


with st.sidebar:
    st.header("⚙️ Experiment Controls")
    n = st.slider("Customers (sample size)", 2000, 40000, 12000, 1000)
    seed = st.number_input("Random seed", 0, 999, 42)
    test_size = st.slider("Holdout fraction", 0.15, 0.40, 0.25, 0.05)
    k = st.slider("Targeting depth k (uplift@k)", 0.05, 0.60, 0.30, 0.05)
    st.caption("Causal ML · Uplift Modeling")

st.title("📈 SalesUplift — Heterogeneous Treatment-Effect Platform")
st.markdown("Estimate **CATE** with T-learner & S-learner metalearners and rank customers "
            "by persuadability using **Qini / AUUC** curves.")

fit, report, curves, data = get_fit(int(n), int(seed), float(test_size))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{data['n_samples']:,}")
c2.metric("Conversion rate", f"{data['positive_rate']:.2%}")
c3.metric("Treatment share", f"{data['treatment_rate']:.1%}")
c4.metric("Observed ATE", f"{data['ate']:.4f}")

tab_q, tab_m, tab_d = st.tabs(["📊 Qini / AUUC", "🔢 Metrics", "🧬 CATE distribution"])

with tab_q:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        frac = np.linspace(0, 1, len(curves["random"]))
        for name in ("random", "T-learner", "S-learner"):
            ax.plot(frac, curves[name], label=name, color=COLORS[name],
                    lw=2 if name != "random" else 1.5, ls="--" if name == "random" else "-")
        ax.set_xlabel("Fraction targeted (sorted by predicted uplift)")
        ax.set_ylabel("Cumulative incremental responses")
        ax.set_title("Qini curve")
        ax.legend(); ax.grid(alpha=0.25)
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ks = np.linspace(0.05, 1.0, 20)
        for name in ("T-learner", "S-learner"):
            vals = [uplift_at_k(fit["y_test"], fit["treatment_test"],
                                fit["uplift_test"][name], k=kk) for kk in ks]
            ax.plot(ks, vals, label=name, color=COLORS[name], lw=2)
        ax.axhline(0, color=COLORS["random"], ls="--", lw=1)
        ax.set_xlabel("Fraction targeted"); ax.set_ylabel("Uplift (treated − control)")
        ax.set_title("Uplift @ k"); ax.legend(); ax.grid(alpha=0.25)
        st.pyplot(fig)

with tab_m:
    rows = []
    for name, m in report["per_model"].items():
        rows.append({"Metalearner": name, **{k: round(v, 4) for k, v in m.items()}})
    st.dataframe(pd.DataFrame(rows).set_index("Metalearner"), use_container_width=True)
    st.caption("Qini coefficient > 0 ⇒ uplift model beats random targeting. "
               "Spearman vs true τ measures CATE-ranking recovery.")

with tab_d:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in ("T-learner", "S-learner"):
            ax.hist(fit["uplift_test"][name], bins=40, alpha=0.55,
                    label=name, color=COLORS[name])
        ax.axvline(0, color=COLORS["random"], ls="--")
        ax.set_xlabel("Predicted CATE"); ax.set_ylabel("Customers (test)")
        ax.set_title("Predicted uplift distribution"); ax.legend(); ax.grid(alpha=0.25)
        st.pyplot(fig)
    with col_b:
        deciles = np.arange(0.1, 1.01, 0.1)
        rows = []
        for name in ("T-learner", "S-learner"):
            row = {"Metalearner": name}
            for dk in deciles:
                row[f"@{int(dk*100)}%"] = round(uplift_at_k(
                    fit["y_test"], fit["treatment_test"], fit["uplift_test"][name], k=dk), 4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Metalearner"), use_container_width=True)
        st.caption("Uplift by targeting decile (top decile should show the largest lift).")

st.markdown("---")
st.markdown("**Method:** Neyman–Rubin potential outcomes · T-learner & S-learner "
            "(Künzel et al. 2019) · Qini (Radcliffe & Surry 2011) · AUUC. "
            "Run `make train` to persist models; `make test` to verify.")
