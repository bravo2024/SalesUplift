from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["customer_tenure_months","avg_purchase_value","purchase_frequency","discount_sensitivity","marketing_channel","campaign_exposure","competitor_activity","seasonality_factor","engagement_score","previous_campaign_response"]
CATEGORICAL_FEATURES = ["marketing_channel","previous_campaign_response"]
NUMERICAL_FEATURES = ["customer_tenure_months","avg_purchase_value","purchase_frequency","discount_sensitivity","campaign_exposure","competitor_activity","seasonality_factor","engagement_score"]
TARGET_NAME = "conversion"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "customer_tenure_months": rng.exponential(scale=24,size=n).clip(1,240).astype(int),
        "avg_purchase_value": rng.lognormal(mean=4,sigma=0.8,size=n).clip(5,500).round(2),
        "purchase_frequency": rng.poisson(lam=3,size=n).clip(0,20),
        "discount_sensitivity": rng.uniform(0,1,size=n).round(3),
        "marketing_channel": rng.choice(["email","social","search","display","affiliate","direct"],size=n,p=[0.25,0.20,0.15,0.15,0.10,0.15]),
        "campaign_exposure": rng.choice([0,1,2,3],size=n,p=[0.2,0.35,0.30,0.15]),
        "competitor_activity": rng.uniform(0,1,size=n).round(3),
        "seasonality_factor": rng.uniform(0,1,size=n).round(3),
        "engagement_score": rng.beta(4,3,size=n).round(3),
        "previous_campaign_response": rng.choice([0,1,2],size=n,p=[0.5,0.35,0.15]),
    })
    ten=np.clip(df["customer_tenure_months"]/240,0,1); val=np.log(df["avg_purchase_value"]+1)/6
    freq=np.clip(df["purchase_frequency"]/20,0,1); disc=df["discount_sensitivity"]
    channel=df["marketing_channel"].map({"email":0,"social":0.2,"search":0.35,"display":0.5,"affiliate":0.7,"direct":1}).values
    exp=np.clip(df["campaign_exposure"]/3,0,1); comp=df["competitor_activity"]
    seas=df["seasonality_factor"]; eng=df["engagement_score"]
    prev=np.clip(df["previous_campaign_response"]/2,0,1)
    log_odds = -2.0 + 0.4*ten + 0.3*val + 0.3*freq + 0.2*disc + 0.2*channel + 0.4*exp - 0.3*comp + 0.1*seas + 0.5*eng + 0.3*prev + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,75)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(conversion=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
