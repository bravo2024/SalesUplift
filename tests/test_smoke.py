from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np; from src.data import make_synthetic; from src.model import train_all_models, cross_validate
def test_data(): d=make_synthetic(500); assert d["X"].shape[0]==500 and 0.1<d["positive_rate"]<0.5
def test_train(): d=make_synthetic(300); b=train_all_models(d); assert len(b["models"])==4
def test_xgb(): d=make_synthetic(300); assert train_all_models(d)["results"]["XGBoost"]["metrics"].get("roc_auc",0)>0.5
def test_cv(): d=make_synthetic(300); cv=cross_validate(d,seed=42,n_folds=3); assert all(s["roc_auc"]["mean"]>0.5 for s in cv.values())
def test_metrics(): d=make_synthetic(200); b=train_all_models(d); m=b["results"]["XGBoost"]["metrics"]; assert all(0<=m[k]<=1 for k in ["accuracy","precision","recall","f1"])
