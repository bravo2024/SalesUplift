from __future__ import annotations
import pickle
from pathlib import Path


def save_model(model, path="models/model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
