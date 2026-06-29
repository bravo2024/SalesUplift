from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import argparse; from src.data import make_synthetic; from src.model import train_all_models, cross_validate
from src.evaluate import save_metrics, print_report; from src.persist import save_model
def main():
    p=argparse.ArgumentParser(); p.add_argument("--n",type=int,default=10000); p.add_argument("--seed",type=int,default=42); p.add_argument("--cv",type=int,default=5)
    a=p.parse_args(); data=make_synthetic(n=a.n,seed=a.seed); print(f"{data['n_samples']:,} samples, conversion rate={data['positive_rate']:.2%}")
    b=train_all_models(data,seed=a.seed); print_report({n:r["metrics"] for n,r in b["results"].items()})
    cv=cross_validate(data,seed=a.seed,n_folds=a.cv)
    for n,s in cv.items(): print(f"  {n:25s} AUC={s['roc_auc']['mean']:.4f} \u00b1{s['roc_auc']['std']:.4f}")
    best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
    save_model({"models":b["models"],"scaler":b["scaler"],"features":b["features"],"best_model":best})
    save_metrics({"holdout":{n:b["results"][n]["metrics"] for n in b["results"]},"cv":cv,"best_model":best})
    print("Saved.")
if __name__=="__main__": main()
