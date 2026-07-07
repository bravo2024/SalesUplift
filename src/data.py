from __future__ import annotations
"""Synthetic data generator for the SalesUplift causal-uplift study.

We simulate a randomized marketing campaign in the Neyman-Rubin potential-outcomes
framework. Each customer has a treatment indicator W (random assignment, ~50%, so
treatment is ignorable / unconfounded), a baseline conversion probability mu0(x)
and a HETEROGENEOUS treatment effect tau(x) = mu1(x) - mu0(x) that depends on
observed customer features. We then draw potential outcomes y0 ~ Bernoulli(mu0),
y1 ~ Bernoulli(mu1) and reveal y = W*y1 + (1-W)*y0. Because tau(x) varies with x,
there is genuine heterogeneous treatment effect (CATE) signal for the metalearners
to recover. The ground-truth tau is retained for offline evaluation.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "customer_tenure_months", "recency_days", "purchase_frequency",
    "avg_order_value", "engagement_score", "discount_sensitivity",
    "loyalty_tier", "channel_preference", "region", "customer_age",
]
CATEGORICAL_FEATURES = ["loyalty_tier", "channel_preference", "region"]
NUMERICAL_FEATURES = [c for c in FEATURE_NAMES if c not in CATEGORICAL_FEATURES]
TREATMENT_COL = "treatment"
TARGET_NAME = "conversion"

_TIER_MAP = {"bronze": 0.15, "silver": 0.35, "gold": 0.60, "platinum": 0.85}


def make_synthetic(n: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # --- customer covariates (distributions chosen to look like real CRM data) ---
    tenure = rng.exponential(scale=28, size=n).clip(1, 240).astype(int)
    recency = rng.exponential(scale=45, size=n).clip(1, 365).astype(int)
    frequency = rng.poisson(lam=4, size=n).clip(0, 30)
    aov = rng.lognormal(mean=4.0, sigma=0.7, size=n).clip(8, 800).round(2)
    engagement = rng.beta(2.5, 3.0, size=n).round(3)
    discount = rng.beta(2.0, 4.0, size=n).round(3)
    tier = rng.choice(["bronze", "silver", "gold", "platinum"], size=n,
                      p=[0.40, 0.30, 0.20, 0.10])
    channel = rng.choice(["email", "sms", "app", "web"], size=n,
                         p=[0.35, 0.20, 0.25, 0.20])
    region = rng.choice(["north", "south", "east", "west", "central"], size=n)
    age = rng.normal(loc=42, scale=12, size=n).clip(18, 80).astype(int)

    # --- randomized treatment assignment (ignorable by design) ---
    w = rng.binomial(1, 0.5, size=n)

    # normalized covariates used only inside the data-generating process
    t_norm = np.clip(tenure / 240.0, 0, 1)
    rec_norm = 1.0 - np.clip(recency / 365.0, 0, 1)   # recent buyers score higher
    freq_norm = np.clip(frequency / 30.0, 0, 1)
    aov_norm = np.clip(np.log(aov + 1) / np.log(801.0), 0, 1)
    eng = engagement
    disc = discount
    tier_val = np.array([_TIER_MAP[t] for t in tier], dtype=float)

    # --- baseline (control) conversion probability mu0(x) ---
    logit0 = (
        -3.0
        + 1.2 * t_norm + 0.8 * rec_norm + 0.9 * freq_norm + 0.6 * aov_norm
        + 1.0 * eng + 0.7 * tier_val
        + rng.normal(0, 0.25, size=n)
    )
    mu0 = 1.0 / (1.0 + np.exp(-logit0))

    # --- heterogeneous CATE tau(x): persuadable vs. sure-thing vs. lost-cause ---
    # High discount-sensitivity + high engagement => strong positive uplift.
    # Lower tiers are more persuadable; very recent buyers show diminishing returns.
    tau = (
        0.08
        + 0.22 * disc
        + 0.18 * eng
        + 0.08 * (1.0 - tier_val)
        - 0.10 * rec_norm
        + 0.06 * freq_norm
        + rng.normal(0, 0.015, size=n)
    )
    tau = np.clip(tau, -0.05, 0.40)

    mu1 = np.clip(mu0 + tau, 0.0, 1.0)
    y0 = rng.binomial(1, mu0).astype(float)
    y1 = rng.binomial(1, mu1).astype(float)
    y = np.where(w == 1, y1, y0).astype(float)

    df = pd.DataFrame({
        "customer_tenure_months": tenure, "recency_days": recency,
        "purchase_frequency": frequency, "avg_order_value": aov,
        "engagement_score": eng, "discount_sensitivity": disc,
        "loyalty_tier": tier, "channel_preference": channel,
        "region": region, "customer_age": age,
        TREATMENT_COL: w, TARGET_NAME: y, "true_tau": tau,
    })

    X = df[FEATURE_NAMES].copy()
    treated = w == 1
    ate = float(y[treated].mean() - y[~treated].mean()) if treated.any() and (~treated).any() else 0.0
    return {
        "X": X,
        "y": y,
        "treatment": w.astype(float),
        "true_tau": tau.astype(float),
        "df": df,
        "features": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "treatment_col": TREATMENT_COL,
        "target_name": TARGET_NAME,
        "n_samples": int(n),
        "positive_rate": float(y.mean()),
        "treatment_rate": float(w.mean()),
        "ate": ate,
    }


HILLSTROM_URL = ("http://www.minethatdata.com/"
                 "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")


def load_hillstrom(cache_dir: str | None = None) -> dict:
    """Hillstrom e-mail experiment, purchase conversion as the outcome.

    64k customers, randomized e-mail vs holdout. Conversion (an actual
    purchase within two weeks) is rare (~0.9%), which makes this a much
    harder uplift target than site visits. No oracle CATE on real data,
    so `true_tau` is None.
    """
    from pathlib import Path

    cache = Path(cache_dir or Path(__file__).parent.parent / "data") / "hillstrom.csv"
    if cache.exists():
        raw = pd.read_csv(cache)
    else:
        raw = pd.read_csv(HILLSTROM_URL)
        cache.parent.mkdir(exist_ok=True)
        raw.to_csv(cache, index=False)

    w = (raw["segment"] != "No E-Mail").astype(int).to_numpy()
    y = raw["conversion"].astype(float).to_numpy()

    features = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
    categorical = ["zip_code", "channel"]

    df = raw[features].copy()
    df["treatment"] = w
    df["conversion"] = y

    treated = w == 1
    ate = float(y[treated].mean() - y[~treated].mean())
    return {
        "X": raw[features].copy(), "y": y, "treatment": w.astype(float),
        "true_tau": None, "df": df, "features": features,
        "categorical_features": categorical,
        "numerical_features": [f for f in features if f not in categorical],
        "treatment_col": "treatment", "target_name": "conversion",
        "n_samples": int(len(df)), "positive_rate": float(y.mean()),
        "treatment_rate": float(w.mean()), "ate": ate,
    }
