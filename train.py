from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse

from src.data import make_synthetic
from src.model import fit_uplift_models
from src.evaluate import uplift_report, save_metrics, print_report
from src.persist import save_model


def main():
    p = argparse.ArgumentParser(description="Train SalesUplift causal-uplift metalearners")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    data = make_synthetic(n=a.n, seed=a.seed)
    print(f"{data['n_samples']:,} customers | conversion={data['positive_rate']:.2%} "
          f"| treatment={data['treatment_rate']:.2%} | observed ATE={data['ate']:.4f}")

    fit = fit_uplift_models(data, seed=a.seed)
    report = uplift_report(fit)
    print_report(report)

    save_model({
        "models": fit["models"],
        "features": fit["features"],
        "categorical_features": fit["categorical_features"],
    })
    save_metrics(report, path="models/metrics.json")
    print("\nSaved models/model.pkl and models/metrics.json")


if __name__ == "__main__":
    main()
