# SalesUplift

> Causal uplift modeling platform with A/B experiment design and power analysis.

Trains four classifiers on synthetic campaign data to predict incremental sales lift from marketing treatment. Dashboard provides full experiment design workflow: treatment/control analysis, uplift model comparison (S-Learner, T-Learner, X-Learner), power analysis with minimum detectable effect, budget allocation optimisation, customer segment targeting with Qini curves, and counterfactual spend simulation.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.778 |
| Gini | 0.557 |
| KS Statistic | 0.476 |
| F1 Score | 0.517 |
| Accuracy | 0.656 |

5-fold CV AUC: 0.770 ± 0.018. Four models compared.

## Features

| Component | What it does |
|---|---|
| **Experiment Designer** | Significance level, MDE, power calculation, sample size estimation |
| **A/B Analysis** | Treatment vs control lift comparison, statistical significance testing |
| **Uplift Models** | S-Learner, T-Learner, X-Learner comparison, Qini curves, uplift by decile |
| **Budget Optimiser** | Campaign budget allocation, ROI by segment, counterfactual scenarios |
| **Power Simulation** | Monte Carlo power curves, sensitivity analysis across effect sizes |

## Repo Structure

```
SalesUplift/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard (780 lines)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic marketing campaign dataset: customer features, treatment indicator, purchase outcome, spend, segment attributes.

## License

MIT
