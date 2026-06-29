from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SalesUplift | Causal Uplift Modeling Platform",
    layout="wide",
    page_icon="📈",
)

# ─── DARK THEME HELPER ────────────────────────────────────────────────────────
def _style(ax, fig=None):
    if fig:
        fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

def _fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("#0f172a")
    axes = np.array(ax).ravel() if np.ndim(ax) > 0 else [ax]
    for a in axes:
        _style(a, fig)
    return fig, ax

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Experiment Controls")
    alpha        = st.slider("Significance Level (α)",  0.01, 0.10, 0.05, 0.005, format="%.3f")
    mde          = st.slider("Minimum Detectable Effect (MDE)", 0.005, 0.10, 0.02, 0.005, format="%.3f")
    budget       = st.number_input("Campaign Budget ($)",   1_000, 500_000, 50_000, 1_000)
    cpc          = st.number_input("Cost per Contact ($)",  0.10,  20.0,    2.0,    0.10)
    aov          = st.number_input("Avg Order Value ($)",   10.0,  500.0,   85.0,   5.0)
    st.caption("SalesUplift | Causal Uplift Modeling Platform")

# ─── SYNTHETIC RCT DATA ───────────────────────────────────────────────────────
@st.cache_data
def make_synthetic(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age                  = rng.integers(18, 71, n).astype(float)
    income_level         = rng.integers(1,  6,  n).astype(float)
    tenure_months        = rng.integers(0,  121, n).astype(float)
    product_usage_score  = rng.uniform(0, 100, n)
    engagement_score     = rng.uniform(0, 100, n)
    region               = rng.choice(["North", "South", "East", "West"], n)
    channel              = rng.choice(["email", "SMS", "push", "mail"], n)
    recency_days         = rng.integers(1, 366, n).astype(float)
    monetary_value       = rng.uniform(10, 5000, n)
    frequency            = rng.integers(1, 53, n).astype(float)

    T = rng.binomial(1, 0.5, n).astype(float)   # 50/50 randomized

    # True heterogeneous CATE
    true_uplift = (
        0.08 * (income_level / 5)
        + 0.06 * (engagement_score / 100)
        - 0.04 * (recency_days / 365)
        + 0.03 * (tenure_months / 120)
        - 0.02 * (1 - product_usage_score / 100)
        - 0.03
    )
    true_uplift = np.clip(true_uplift, -0.15, 0.30)

    # Base (control) conversion probability
    base_p = np.clip(
        0.05
        + 0.10 * (engagement_score / 100)
        + 0.06 * (income_level / 5)
        + 0.04 * (product_usage_score / 100)
        - 0.03 * (recency_days / 365),
        0.02, 0.80,
    )

    treated_p  = np.clip(base_p + true_uplift * T, 0.01, 0.99)
    purchased  = rng.binomial(1, treated_p).astype(float)
    revenue_30d = np.where(
        purchased == 1,
        rng.uniform(10, 170, n) * (1 + 0.3 * T * true_uplift),
        0.0,
    )

    return pd.DataFrame(dict(
        age=age, income_level=income_level, tenure_months=tenure_months,
        product_usage_score=product_usage_score, engagement_score=engagement_score,
        region=region, channel=channel, recency_days=recency_days,
        monetary_value=monetary_value, frequency=frequency,
        T=T, purchased_within_30d=purchased, revenue_30d=revenue_30d,
        true_uplift=true_uplift,
    ))

# ─── NUMPY SGD LOGISTIC REGRESSION ───────────────────────────────────────────
NUMERIC_FEATS = [
    "age", "income_level", "tenure_months", "product_usage_score",
    "engagement_score", "recency_days", "monetary_value", "frequency",
]

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def _logreg_fit(X, y, lr=0.05, epochs=300, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(X.shape[1]) * 0.01
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        p   = _sigmoid(X @ w + b)
        err = p - y
        w  -= lr / n * (X.T @ err)
        b  -= lr / n * err.sum()
    return w, b

def _predict(X, w, b):
    return _sigmoid(X @ w + b)

@st.cache_resource
def train_uplift_models(_df: pd.DataFrame):
    X = _df[NUMERIC_FEATS].values.astype(float)
    T = _df["T"].values.astype(float)
    Y = _df["purchased_within_30d"].values.astype(float)

    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    Xn  = (X - mu) / sig

    # S-Learner — single model trained on (X, T)
    XT   = np.column_stack([Xn, T])
    w_s, b_s = _logreg_fit(XT, Y, lr=0.1, epochs=400, seed=0)
    XT1  = np.column_stack([Xn, np.ones(len(T))])
    XT0  = np.column_stack([Xn, np.zeros(len(T))])
    mu1_s = _predict(XT1, w_s, b_s)
    mu0_s = _predict(XT0, w_s, b_s)
    tau_s = mu1_s - mu0_s

    # T-Learner — separate models per arm
    mask1 = T == 1;  mask0 = T == 0
    w1, b1 = _logreg_fit((X[mask1] - mu) / sig, Y[mask1], lr=0.1, epochs=400, seed=1)
    w0, b0 = _logreg_fit((X[mask0] - mu) / sig, Y[mask0], lr=0.1, epochs=400, seed=2)
    mu1_t  = _predict(Xn, w1, b1)
    mu0_t  = _predict(Xn, w0, b0)
    tau_t  = mu1_t - mu0_t

    return dict(tau_s=tau_s, mu1_s=mu1_s, mu0_s=mu0_s,
                tau_t=tau_t, mu1_t=mu1_t, mu0_t=mu0_t)

# ─── VECTORIZED EVALUATION HELPERS ───────────────────────────────────────────
def qini_curve(tau, T, Y):
    """Vectorised Qini curve: O(n log n) sort then O(n) cumsum."""
    order    = np.argsort(-tau)
    T_o, Y_o = T[order], Y[order]
    N_T = T_o.sum();  N_C = (1 - T_o).sum()
    cum_T = np.cumsum(Y_o * T_o)
    cum_C = np.cumsum(Y_o * (1 - T_o))
    qvals = cum_T / (N_T + 1e-9) - cum_C / (N_C + 1e-9)
    fracs = np.linspace(0, 1, len(qvals) + 1)
    return fracs, np.concatenate([[0.0], qvals])

def uplift_curve(tau, T, Y, n_pts=101):
    order    = np.argsort(-tau)
    T_o, Y_o = T[order], Y[order]
    n        = len(T_o)
    pcts     = np.linspace(0, 1, n_pts)
    uvals    = [0.0]
    for pct in pcts[1:]:
        k    = max(1, int(pct * n))
        sT   = T_o[:k];  sY = Y_o[:k]
        u    = (sY[sT == 1].mean() - sY[sT == 0].mean()
                if sT.sum() > 0 and (1 - sT).sum() > 0 else 0.0)
        uvals.append(u)
    return pcts, np.array(uvals)

# ─── LOAD DATA & MODELS ───────────────────────────────────────────────────────
df      = make_synthetic(n=20_000)
models  = train_uplift_models(df)
tau_s   = models["tau_s"];   tau_t  = models["tau_t"]
mu1_s   = models["mu1_s"];   mu0_s  = models["mu0_s"]
mu1_t   = models["mu1_t"];   mu0_t  = models["mu0_t"]

T_arr   = df["T"].values.astype(float)
Y_arr   = df["purchased_within_30d"].values.astype(float)
N       = len(df)

# ─── FREQUENTIST A/B STATS ────────────────────────────────────────────────────
n1  = T_arr.sum();         n0  = N - n1
p1  = Y_arr[T_arr == 1].mean();  p0 = Y_arr[T_arr == 0].mean()
p_pool  = Y_arr.mean()
se      = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
z_stat  = (p1 - p0) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
ci_lo   = (p1 - p0) - 1.96 * np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
ci_hi   = (p1 - p0) + 1.96 * np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
ate     = p1 - p0

# ─── QINI / AUUC ─────────────────────────────────────────────────────────────
fracs_s, qini_s = qini_curve(tau_s, T_arr, Y_arr)
fracs_t, qini_t = qini_curve(tau_t, T_arr, Y_arr)
auuc_s = float(np.trapz(qini_s, fracs_s))
auuc_t = float(np.trapz(qini_t, fracs_t))

# ─── ROI SWEEP ────────────────────────────────────────────────────────────────
order_by_tau = np.argsort(-tau_s)
reach_pcts   = np.linspace(0, 1, 101)
inc_rev_arr  = np.array([
    tau_s[order_by_tau[:max(1, int(p * N))]].clip(0).sum() * aov
    for p in reach_pcts
])
cost_arr     = reach_pcts * N * cpc
roi_arr      = np.where(cost_arr > 0, (inc_rev_arr - cost_arr) / cost_arr * 100, 0.0)
opt_idx      = int(np.argmax(roi_arr))
optimal_pct  = reach_pcts[opt_idx] * 100
optimal_roi  = roi_arr[opt_idx]

# ─── HEADER KPIs ──────────────────────────────────────────────────────────────
st.title("📈 SalesUplift — Causal Uplift Modeling Platform")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Customers (N)", f"{N:,}")
k2.metric("ATE (Lift)",     f"{ate:+.4f}")
k3.metric("p-value",        f"{p_value:.4f}",
          delta="Significant ✅" if p_value < alpha else "Not sig ⚠️")
k4.metric("Qini (S-Learner)", f"{auuc_s:.4f}")
k5.metric("Optimal Reach",    f"{optimal_pct:.1f}%")
k6.metric("Max ROI",          f"{optimal_roi:.1f}%")
st.divider()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Experiment Design & Data",
    "📊 A/B Test Analysis",
    "🎯 Uplift Modeling (S-Learner & T-Learner)",
    "📈 Uplift Evaluation Metrics",
    "💰 Campaign ROI Optimization",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPERIMENT DESIGN & DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Randomized Controlled Trial (RCT) — Synthetic Dataset (N=20,000)")
    st.latex(r"\tau(x) = \mathbb{E}\!\left[Y(1) - Y(0) \mid X = x\right]")

    c_info, c_cards = st.columns([3, 1])
    with c_info:
        st.info(
            "**20,000 customers** | Treatment T=1 (discount/promo) vs T=0 (control) | **50/50 randomized**\n\n"
            "True uplift is **heterogeneous**: large for high-income, high-engagement, low-recency customers; "
            "near-zero or negative for low-engagement churners (Sleeping Dogs)."
        )
    with c_cards:
        st.metric("Conv Rate (T=1)", f"{p1:.3%}")
        st.metric("Conv Rate (T=0)", f"{p0:.3%}")
        st.metric("Naïve ATE",       f"{ate:+.4f}")

    st.dataframe(df.drop(columns=["true_uplift"]).head(100),
                 use_container_width=True, height=220)

    # ── Covariate Balance ────────────────────────────────────────────────────
    st.subheader("Covariate Balance — Standardized Mean Difference (SMD)")
    st.latex(r"\mathrm{SMD} = \frac{\mu_1 - \mu_0}{\sigma_{\mathrm{pooled}}}")

    bal_rows = []
    for feat in NUMERIC_FEATS:
        x1f = df.loc[df["T"] == 1, feat].values
        x0f = df.loc[df["T"] == 0, feat].values
        sp  = np.sqrt((x1f.std()**2 + x0f.std()**2) / 2) + 1e-9
        smd = (x1f.mean() - x0f.mean()) / sp
        bal_rows.append({
            "Feature":     feat,
            "Mean (T=1)":  f"{x1f.mean():.3f}",
            "Mean (T=0)":  f"{x0f.mean():.3f}",
            "SMD":         f"{smd:.4f}",
            "Balanced?":   "✅" if abs(smd) < 0.10 else "⚠️",
        })
    st.dataframe(pd.DataFrame(bal_rows), use_container_width=True)

    # ── Outcome Distribution ─────────────────────────────────────────────────
    st.subheader("Outcome Distribution: Treatment vs Control")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = _fig(figsize=(6, 4))
        r1 = df.loc[df["T"] == 1, "revenue_30d"].values
        r0 = df.loc[df["T"] == 0, "revenue_30d"].values
        ax.hist(r1[r1 > 0], bins=45, alpha=0.6, color="#22c55e",
                label="T=1 (Treatment)", density=True)
        ax.hist(r0[r0 > 0], bins=45, alpha=0.6, color="#f43f5e",
                label="T=0 (Control)",   density=True)
        ax.set_title("30-Day Revenue (purchasers only)", color="white")
        ax.set_xlabel("Revenue ($)")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    with col_b:
        fig, ax = _fig(figsize=(6, 4))
        cats   = ["Control (T=0)", "Treatment (T=1)"]
        convs  = [p0, p1]
        colors = ["#f43f5e", "#22c55e"]
        bars   = ax.bar(cats, convs, color=colors, width=0.5)
        for bar, val in zip(bars, convs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.002, f"{val:.3%}",
                    ha="center", color="white", fontsize=11, fontweight="bold")
        ax.set_title("30-Day Purchase Rate by Group", color="white")
        ax.set_ylabel("Conversion Rate")
        ax.set_ylim(0, max(convs) * 1.3)
        ax.grid(True, alpha=0.2, axis="y")
        st.pyplot(fig)

    # ── Customer Segments Pie ────────────────────────────────────────────────
    st.subheader("Customer Segments — Uplift Quadrants")
    persuadables_n  = int(((tau_s > 0) & (mu0_s < 0.5)).sum())
    sure_things_n   = int(((mu1_s > 0.5) & (mu0_s > 0.5)).sum())
    sleeping_dogs_n = int((tau_s < 0).sum())
    lost_causes_n   = N - persuadables_n - sure_things_n - sleeping_dogs_n

    fig, ax = _fig(figsize=(5, 4))
    ax.pie(
        [persuadables_n, sure_things_n, lost_causes_n, sleeping_dogs_n],
        labels=["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"],
        colors=["#22c55e", "#3b82f6", "#94a3b8", "#f43f5e"],
        autopct="%1.1f%%", startangle=140,
        textprops={"color": "white", "fontsize": 9},
    )
    ax.set_title("S-Learner Customer Segments", color="white")
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — A/B TEST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Frequentist A/B Testing — Two-Proportion z-Test")
    st.latex(
        r"z = \frac{\hat{p}_1 - \hat{p}_0}"
        r"{\sqrt{\hat{p}(1-\hat{p})\!\left(\dfrac{1}{n_1}+\dfrac{1}{n_0}\right)}}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p̂₁ (Treatment)", f"{p1:.4f}")
    c2.metric("p̂₀ (Control)",   f"{p0:.4f}")
    c3.metric("z-statistic",     f"{z_stat:.4f}")
    c4.metric("p-value",         f"{p_value:.4f}",
              delta="Significant ✅" if p_value < alpha else "Not sig ⚠️")

    st.markdown(f"**95% Confidence Interval for (p₁−p₀):** [{ci_lo:.4f}, {ci_hi:.4f}]")
    st.latex(
        r"\mathrm{CI}_{95\%} = (\hat{p}_1 - \hat{p}_0)"
        r"\pm 1.96\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_0(1-\hat{p}_0)}{n_0}}"
    )

    # ── Sample Size ──────────────────────────────────────────────────────────
    st.subheader("Required Sample Size — Power Analysis (β = 0.20)")
    st.latex(
        r"n = \frac{(z_{\alpha/2}+z_{\beta})^2 \cdot 2\hat{p}(1-\hat{p})}{\delta^2}"
    )
    z_a2 = stats.norm.ppf(1 - alpha / 2)
    z_b  = stats.norm.ppf(0.80)
    n_req = int(np.ceil((z_a2 + z_b)**2 * 2 * p_pool * (1 - p_pool) / mde**2))
    s1, s2, s3 = st.columns(3)
    s1.metric("MDE (δ)",           f"{mde:.3f}")
    s2.metric("Required n per arm", f"{n_req:,}")
    s3.metric("Total N required",   f"{2 * n_req:,}")

    # ── Bayesian A/B ─────────────────────────────────────────────────────────
    st.subheader("Bayesian A/B — Beta-Binomial Posterior")
    st.latex(r"\theta \sim \mathrm{Beta}(\alpha + s,\; \beta + (n - s))")
    st.latex(
        r"P(\theta_1 > \theta_0) = \int_0^1"
        r"\mathrm{Beta}(\theta_1;\,\alpha_1,\beta_1)\cdot"
        r"P(\theta_0 < \theta_1;\,\alpha_0,\beta_0)\,d\theta_1"
    )
    rng_mc   = np.random.default_rng(99)
    s_t1     = int(Y_arr[T_arr == 1].sum())
    s_t0     = int(Y_arr[T_arr == 0].sum())
    samp_t1  = rng_mc.beta(1 + s_t1, 1 + (int(n1) - s_t1), size=10_000)
    samp_t0  = rng_mc.beta(1 + s_t0, 1 + (int(n0) - s_t0), size=10_000)
    prob_win = float((samp_t1 > samp_t0).mean())
    diff_mc  = samp_t1 - samp_t0

    b1, b2, b3 = st.columns(3)
    b1.metric("P(Treatment > Control)", f"{prob_win:.4f}")
    b2.metric("Posterior Expected Lift", f"{diff_mc.mean():.4f}")
    b3.metric("95% Credible Interval",
              f"[{np.percentile(diff_mc, 2.5):.4f}, {np.percentile(diff_mc, 97.5):.4f}]")

    fig, ax = _fig(figsize=(9, 4))
    ax.hist(samp_t1, bins=80, alpha=0.6, color="#22c55e",
            label="θ₁ posterior (Treatment)", density=True)
    ax.hist(samp_t0, bins=80, alpha=0.6, color="#f43f5e",
            label="θ₀ posterior (Control)",   density=True)
    ax.set_title("Bayesian Posterior Distributions — 10,000 Monte Carlo Draws", color="white")
    ax.set_xlabel("Conversion Rate θ")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    # ── CUPED ────────────────────────────────────────────────────────────────
    st.subheader("CUPED — Variance Reduction via Pre-Experiment Covariate")
    st.latex(r"Y_{\mathrm{CUPED}} = Y - \theta\,(X_{\mathrm{pre}} - \bar{X}_{\mathrm{pre}})")
    st.latex(r"\theta = \frac{\mathrm{Cov}(Y,\,X_{\mathrm{pre}})}{\mathrm{Var}(X_{\mathrm{pre}})}")

    X_pre   = df["engagement_score"].values
    theta_c = np.cov(Y_arr.astype(float), X_pre)[0, 1] / np.var(X_pre)
    Y_cuped = Y_arr.astype(float) - theta_c * (X_pre - X_pre.mean())
    var_red = (np.var(Y_arr) - np.var(Y_cuped)) / np.var(Y_arr) * 100

    cu1, cu2, cu3 = st.columns(3)
    cu1.metric("θ (theta)",          f"{theta_c:.6f}")
    cu2.metric("Variance Reduction",  f"{var_red:.2f}%")
    cu3.metric("CUPED ATE",
               f"{Y_cuped[T_arr==1].mean()-Y_cuped[T_arr==0].mean():+.4f}")

    # ── Sequential Testing ────────────────────────────────────────────────────
    st.subheader("Sequential Testing — O'Brien-Fleming Alpha-Spending Boundary")
    st.latex(
        r"\alpha^*(t) \approx 2\!\left(1 - \Phi\!\left(\frac{z_{\alpha/2}}{\sqrt{t}}\right)\right)"
    )
    t_fracs      = np.linspace(0.01, 1.0, 200)
    alpha_spent  = 2 * (1 - stats.norm.cdf(z_a2 / np.sqrt(t_fracs)))

    fig, ax = _fig(figsize=(9, 3))
    ax.plot(t_fracs * 100, alpha_spent, color="#f59e0b", lw=2,
            label="O'Brien-Fleming boundary")
    ax.axhline(alpha, color="#f43f5e", ls="--", lw=1.5, label=f"α = {alpha:.3f}")
    ax.fill_between(t_fracs * 100, alpha_spent, alpha=0.12, color="#f59e0b")
    ax.set_xlabel("Information Fraction (%)")
    ax.set_ylabel("Cumulative α Spent")
    ax.set_title("Sequential Testing — Alpha Spending Boundary", color="white")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — UPLIFT MODELING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Causal Uplift Modeling — S-Learner & T-Learner (NumPy SGD Only)")
    st.latex(r"\tau(x) = \mathbb{E}\!\left[Y(1) - Y(0) \mid X = x\right]")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**S-Learner**")
        st.latex(r"\hat{\tau}_S(x) = \hat{f}(x,\,T{=}1) - \hat{f}(x,\,T{=}0)")
        st.info("One logistic regression trained on (X, T) → Y. "
                "T is treated as just another input feature.")
    with col_r:
        st.markdown("**T-Learner**")
        st.latex(r"\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)")
        st.info("Separate logistic regressions: μ₁ on treated group, "
                "μ₀ on control group. CATE = difference of predictions.")

    # ── ITE Distribution ─────────────────────────────────────────────────────
    st.subheader("Individual Treatment Effect (ITE) Distributions")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    _style(ax1); _style(ax2)

    ax1.hist(tau_s, bins=60, color="#3b82f6", alpha=0.85, edgecolor="#1e3a5f")
    ax1.axvline(tau_s.mean(), color="#fbbf24", lw=2, ls="--",
                label=f"Mean = {tau_s.mean():.4f}")
    ax1.axvline(0, color="#94a3b8", lw=1, ls=":")
    ax1.set_title("S-Learner  τ̂(x)", color="white")
    ax1.set_xlabel("Estimated Uplift τ̂")
    ax1.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax1.grid(True, alpha=0.2)

    ax2.hist(tau_t, bins=60, color="#a855f7", alpha=0.85, edgecolor="#3b0764")
    ax2.axvline(tau_t.mean(), color="#fbbf24", lw=2, ls="--",
                label=f"Mean = {tau_t.mean():.4f}")
    ax2.axvline(0, color="#94a3b8", lw=1, ls=":")
    ax2.set_title("T-Learner  τ̂(x)", color="white")
    ax2.set_xlabel("Estimated Uplift τ̂")
    ax2.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)

    # ── 4-Segment Classification ──────────────────────────────────────────────
    st.subheader("4-Segment Uplift Classification (T-Learner)")
    st.latex(
        r"\text{Persuadables}: \hat{\tau}>0,\;\hat{\mu}_0<0.5 \qquad"
        r"\text{Sure Things}: \hat{\mu}_1>0.5,\;\hat{\mu}_0>0.5"
    )
    st.latex(
        r"\text{Lost Causes}: \hat{\mu}_1<0.5,\;\hat{\mu}_0<0.5,\;\hat{\tau}\geq 0 \qquad"
        r"\text{Sleeping Dogs}: \hat{\tau}<0"
    )

    m_p  = (tau_t > 0) & (mu0_t < 0.5)
    m_st = (mu1_t > 0.5) & (mu0_t > 0.5)
    m_sd = tau_t < 0
    m_lc = ~m_p & ~m_st & ~m_sd

    seg_data = [
        ("Persuadables", m_p,  "✅ Target", "#22c55e"),
        ("Sure Things",  m_st, "🔵 Optional","#3b82f6"),
        ("Lost Causes",  m_lc, "🔴 Skip",   "#94a3b8"),
        ("Sleeping Dogs",m_sd, "⛔ Avoid",  "#f43f5e"),
    ]
    seg_rows = []
    for name, mask, action, _ in seg_data:
        cnt = mask.sum()
        seg_rows.append({
            "Segment":        name,
            "Count":          f"{cnt:,}",
            "% of Customers": f"{cnt/N:.1%}",
            "Avg τ̂":         f"{tau_t[mask].mean():.4f}" if cnt > 0 else "—",
            "Recommendation": action,
        })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)

    col_pie, col_sc = st.columns(2)
    with col_pie:
        fig, ax = _fig(figsize=(5, 4))
        ax.pie(
            [m_p.sum(), m_st.sum(), m_lc.sum(), m_sd.sum()],
            labels=["Persuadables","Sure Things","Lost Causes","Sleeping Dogs"],
            colors=["#22c55e","#3b82f6","#94a3b8","#f43f5e"],
            autopct="%1.1f%%", startangle=140,
            textprops={"color":"white","fontsize":8},
        )
        ax.set_title("T-Learner Segment Proportions", color="white")
        st.pyplot(fig)
    with col_sc:
        fig, ax = _fig(figsize=(5, 4))
        rng_sc = np.random.default_rng(7)
        idx_sc = rng_sc.choice(N, 2000, replace=False)
        sc = ax.scatter(
            mu0_t[idx_sc], mu1_t[idx_sc],
            c=tau_t[idx_sc], cmap="RdYlGn", alpha=0.5, s=8,
        )
        ax.axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0.5, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("μ̂₀(x) — Control Probability")
        ax.set_ylabel("μ̂₁(x) — Treatment Probability")
        ax.set_title("T-Learner: μ₁ vs μ₀  (coloured by τ̂)", color="white")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("τ̂ (uplift)", color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — UPLIFT EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Uplift Model Evaluation — Qini Curves, AUUC & Decile Analysis")
    st.latex(
        r"\mathrm{Qini}(k) = \frac{R_{T,k}}{N_T} - \frac{R_{C,k}}{N_C}"
    )
    st.latex(
        r"\mathrm{AUUC} = \int_0^1 \mathrm{UpliftCurve}(k)\,dk"
        r"\approx \sum_k \hat\tau_k \cdot \Delta k"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUUC — S-Learner", f"{auuc_s:.5f}")
    m2.metric("AUUC — T-Learner", f"{auuc_t:.5f}")
    m3.metric("Mean τ̂ (S)",      f"{tau_s.mean():.4f}")
    m4.metric("Mean τ̂ (T)",      f"{tau_t.mean():.4f}")

    # ── Qini & Uplift Curves ──────────────────────────────────────────────────
    fig, (ax_q, ax_u) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f172a")
    _style(ax_q); _style(ax_u)

    # Subsample to 2000 points for plotting
    pts = np.linspace(0, len(fracs_s) - 1, 2000).astype(int)
    ax_q.plot(fracs_s[pts]*100, qini_s[pts], color="#3b82f6", lw=2,
              label=f"S-Learner (AUUC={auuc_s:.4f})")
    ax_q.plot(fracs_t[pts]*100, qini_t[pts], color="#a855f7", lw=2,
              label=f"T-Learner (AUUC={auuc_t:.4f})")
    ax_q.axhline(0, color="#94a3b8", ls="--", lw=1)
    ax_q.set_xlabel("Population Reached (%)")
    ax_q.set_ylabel("Qini Value")
    ax_q.set_title("Qini Curves", color="white")
    ax_q.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax_q.grid(True, alpha=0.2)

    pcts_s, uc_s = uplift_curve(tau_s, T_arr, Y_arr)
    pcts_t, uc_t = uplift_curve(tau_t, T_arr, Y_arr)
    ax_u.plot(pcts_s*100, uc_s, color="#3b82f6", lw=2, label="S-Learner")
    ax_u.plot(pcts_t*100, uc_t, color="#a855f7", lw=2, label="T-Learner")
    ax_u.axhline(ate, color="#fbbf24", ls="--", lw=1.5, label=f"Naïve ATE={ate:.4f}")
    ax_u.axhline(0,   color="#94a3b8", ls=":",  lw=1)
    ax_u.set_xlabel("Population Reached (%)")
    ax_u.set_ylabel("Actual Uplift (Y|T=1 − Y|T=0)")
    ax_u.set_title("Uplift Curves (Gains)", color="white")
    ax_u.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax_u.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Decile Table ─────────────────────────────────────────────────────────
    st.subheader("Decile Analysis — Ranked by τ̂ (S-Learner, D1=Best)")
    order_s_dec  = np.argsort(-tau_s)
    n_per_decile = N // 10
    dec_rows = []
    for d in range(10):
        idx  = order_s_dec[d * n_per_decile: (d+1) * n_per_decile]
        sT   = T_arr[idx]; sY = Y_arr[idx]
        t_n  = sT.sum(); c_n = (1-sT).sum()
        up   = (sY[sT==1].mean() - sY[sT==0].mean()
                if t_n > 0 and c_n > 0 else np.nan)
        dec_rows.append({
            "Decile":        f"D{d+1}",
            "Avg τ̂":        f"{tau_s[idx].mean():.4f}",
            "n Treated":     int(t_n),
            "n Control":     int(c_n),
            "Conv (T=1)":    f"{sY[sT==1].mean():.3%}" if t_n > 0 else "—",
            "Conv (T=0)":    f"{sY[sT==0].mean():.3%}" if c_n > 0 else "—",
            "Actual Uplift": f"{up:.4f}" if not np.isnan(up) else "—",
        })
    st.dataframe(pd.DataFrame(dec_rows), use_container_width=True)

    # ── Decile Bar Chart ──────────────────────────────────────────────────────
    fig, ax = _fig(figsize=(10, 4))
    vals   = [float(r["Actual Uplift"]) if r["Actual Uplift"] != "—" else 0.0
              for r in dec_rows]
    colors = ["#22c55e" if v > 0 else "#f43f5e" for v in vals]
    ax.bar(range(1, 11), vals, color=colors, edgecolor="#0f172a")
    ax.axhline(0, color="white", lw=0.8, ls="--")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("Decile  (D1 = highest τ̂)")
    ax.set_ylabel("Actual Uplift")
    ax.set_title("Actual Uplift per Decile — Should Decrease D1 → D10", color="white")
    ax.grid(True, alpha=0.2, axis="y")
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CAMPAIGN ROI OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Campaign ROI Optimization — Optimal Targeting Strategy")
    st.latex(
        r"\mathrm{ROI}(k) = "
        r"\frac{\displaystyle\sum_{i\in\mathrm{top\text{-}k}}\hat\tau(x_i)\cdot\mathrm{AOV}"
        r"\ -\ k\cdot n\cdot c_{\mathrm{contact}}}"
        r"{k\cdot n\cdot c_{\mathrm{contact}}}\times 100\%"
    )

    # ── ROI & Revenue vs Cost ─────────────────────────────────────────────────
    fig, (ax_roi, ax_rev) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f172a")
    _style(ax_roi); _style(ax_rev)

    ax_roi.plot(reach_pcts*100, roi_arr, color="#22c55e", lw=2.5, label="ROI (%)")
    ax_roi.axvline(optimal_pct, color="#fbbf24", lw=2, ls="--",
                   label=f"Optimal k* = {optimal_pct:.1f}%")
    ax_roi.axhline(0, color="#94a3b8", lw=1, ls=":")
    ax_roi.set_xlabel("Customers Targeted (%)")
    ax_roi.set_ylabel("ROI (%)")
    ax_roi.set_title("ROI vs Reach Percentage", color="white")
    ax_roi.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax_roi.grid(True, alpha=0.2)

    ax_rev.plot(reach_pcts*100, inc_rev_arr/1e3, color="#3b82f6", lw=2,
                label="Incremental Revenue ($K)")
    ax_rev.plot(reach_pcts*100, cost_arr/1e3,    color="#f43f5e", lw=2, ls="--",
                label="Campaign Cost ($K)")
    ax_rev.fill_between(reach_pcts*100, inc_rev_arr/1e3, cost_arr/1e3,
                        where=(inc_rev_arr > cost_arr), alpha=0.18,
                        color="#22c55e", label="Profit zone")
    ax_rev.fill_between(reach_pcts*100, inc_rev_arr/1e3, cost_arr/1e3,
                        where=(inc_rev_arr <= cost_arr), alpha=0.18,
                        color="#f43f5e", label="Loss zone")
    ax_rev.set_xlabel("Customers Targeted (%)")
    ax_rev.set_ylabel("Amount ($K)")
    ax_rev.set_title("Incremental Revenue vs Campaign Cost", color="white")
    ax_rev.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax_rev.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Optimal Reach (k*)",      f"{optimal_pct:.1f}%")
    o2.metric("Max ROI",                  f"{optimal_roi:.1f}%")
    o3.metric("Incr. Revenue at k*",      f"${inc_rev_arr[opt_idx]:,.0f}")
    o4.metric("Campaign Cost at k*",      f"${cost_arr[opt_idx]:,.0f}")

    # ── Budget Allocator ──────────────────────────────────────────────────────
    st.subheader("Budget Allocator")
    max_k     = min(budget / (cpc * N), 1.0)
    bk_idx    = min(int(max_k * 100), len(roi_arr) - 1)
    bk_roi    = roi_arr[bk_idx]
    bk_rev    = inc_rev_arr[bk_idx]

    ba1, ba2, ba3 = st.columns(3)
    ba1.metric("Budget",                   f"${budget:,.0f}")
    ba2.metric("Max Reach (budget-capped)", f"{max_k*100:.1f}%")
    ba3.metric("Expected ROI at budget",    f"{bk_roi:.1f}%")

    st.info(
        f"With **${budget:,}** at **${cpc:.2f}** / contact you can reach "
        f"**{max_k*100:.1f}%** of the base (**{int(max_k*N):,} customers**). "
        f"Expected incremental revenue: **${bk_rev:,.0f}**."
    )

    # ── Revenue Impact Table ──────────────────────────────────────────────────
    st.subheader("Revenue Impact at Fixed Reach Levels")
    impact_rows = []
    for pct in [0.10, 0.20, 0.30]:
        k      = int(pct * N)
        top_k  = order_by_tau[:k]
        ir     = tau_s[top_k].clip(0).sum() * aov
        cost   = k * cpc
        roi_p  = (ir - cost) / max(cost, 1) * 100
        rand_ir= Y_arr.mean() * k * aov * ate           # random targeting baseline
        savings= ir - max(rand_ir, 0)
        impact_rows.append({
            "Reach":              f"{pct:.0%}",
            "Customers Targeted": f"{k:,}",
            "Incr. Revenue":      f"${ir:,.0f}",
            "Campaign Cost":      f"${cost:,.0f}",
            "ROI":                f"{roi_p:.1f}%",
            "vs Random Targeting":f"+${savings:,.0f}",
        })
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True)

    # ── Microsegment Recommendations ──────────────────────────────────────────
    st.subheader("Microsegment Targeting Recommendations")
    seg_rec_rows = []
    for name, mask, action, icon in [
        ("Persuadables",  m_p,  "✅ Target — highest incremental ROI",   "🟢"),
        ("Sure Things",   m_st, "🔵 Optional — converts regardless",     "🔵"),
        ("Lost Causes",   m_lc, "🔴 Skip — minimal response to promo",  "🟡"),
        ("Sleeping Dogs", m_sd, "⛔ Avoid — promotion backfires",        "🔴"),
    ]:
        cnt = mask.sum()
        if cnt == 0:
            continue
        avg_tau  = tau_t[mask].mean()
        avg_rev  = avg_tau * aov
        seg_rec_rows.append({
            "Segment":              f"{icon} {name}",
            "Count":                f"{cnt:,}",
            "% of Base":            f"{cnt/N:.1%}",
            "Avg τ̂ (T-Learner)":   f"{avg_tau:.4f}",
            "Expected Rev/Contact": f"${avg_rev:.2f}",
            "Recommendation":       action,
        })
    st.dataframe(pd.DataFrame(seg_rec_rows), use_container_width=True)

    # ── Savings vs Random Targeting ───────────────────────────────────────────
    st.subheader("Uplift-Based Targeting vs Random Selection — Incremental Value")
    st.latex(
        r"\Delta\mathrm{Revenue}(k) = "
        r"\mathrm{Revenue}_{\mathrm{uplift}}(k) - \mathrm{Revenue}_{\mathrm{random}}(k)"
    )
    # Random baseline: uniform mix means avg τ̂ across all customers
    rand_inc = reach_pcts * tau_s.clip(0).sum() * aov        # linear scale
    savings  = inc_rev_arr - rand_inc

    fig, ax = _fig(figsize=(10, 4))
    ax.plot(reach_pcts*100, savings/1e3, color="#f59e0b", lw=2.5,
            label="Savings vs random ($K)")
    ax.axhline(0, color="#94a3b8", lw=1, ls="--")
    ax.axvline(optimal_pct, color="#fbbf24", lw=2, ls=":",
               label=f"Optimal k* = {optimal_pct:.1f}%")
    ax.fill_between(reach_pcts*100, savings/1e3, 0,
                    where=(savings > 0), alpha=0.18, color="#22c55e")
    ax.fill_between(reach_pcts*100, savings/1e3, 0,
                    where=(savings <= 0), alpha=0.18, color="#f43f5e")
    ax.set_xlabel("Customers Targeted (%)")
    ax.set_ylabel("Incremental Value over Random ($K)")
    ax.set_title("Uplift-Ranked Targeting: Value Gained vs Random Baseline", color="white")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
