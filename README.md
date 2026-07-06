# SalesUplift — Causal Uplift Modeling

Estimate the **heterogeneous treatment effect (CATE)** of a marketing campaign and
rank customers by *persuadability* so budget is spent on the people who actually
respond to treatment — not on sure-thing buyers or lost causes.

This is a **causal-inference / uplift** project, **not** a classifier. There is no
single "conversion probability" to predict; the target is the *incremental* effect of
treatment, which is never observed for any individual (the fundamental problem of
causal inference).

## Methodology

### Potential-outcomes data model
Each customer `i` has covariates `X_i`, a randomized treatment `W_i ~ Bernoulli(0.5)`
(ignorable by design), and potential outcomes `y0_i, y1_i ~ Bernoulli(mu0(X_i)), Bernoulli(mu1(X_i))`.
We observe `y_i = W_i·y1_i + (1−W_i)·y0_i`. The conditional average treatment effect
is `τ(x) = E[y1 − y0 | X = x] = mu1(x) − mu0(x)`, and it is **heterogeneous** — it
grows with discount-sensitivity and engagement and shrinks for already-recent buyers.

### Metalearners (Künzel et al. 2019)
- **T-learner** — fit `μ̂1` on treated units and `μ̂0` on control units separately
  (gradient-boosted trees); `CATE(x) = μ̂1(x) − μ̂0(x)`.
- **S-learner** — fit a single model `p̂(y | X, W)` with treatment as a feature;
  `CATE(x) = p̂(x, W=1) − p̂(x, W=0)`.

### Evaluation metrics (from scratch)
- **Qini curve** — sort by predicted uplift; `q(f) = Y_t(f) − Y_c(f)·N_t(f)/N_c(f)`,
  the incremental responses captured among the top-`f` fraction targeted.
- **Qini coefficient** — area between the model Qini curve and the random
  (straight-line) baseline; > 0 means the model beats random targeting.
- **AUUC** — area under the uplift curve (treated−control conversion gap among the
  targeted fraction).
- **uplift@k** — conversion-rate gap within the top-k% by predicted uplift.
- **Expected response lift** — expected incremental conversions from treating top-k%.

## Project layout
```
src/data.py      synthetic potential-outcomes data with heterogeneous CATE
src/model.py     T-learner & S-learner metalearners (LightGBM base learners)
src/core.py      Qini, Qini coefficient, AUUC, uplift@k, ATE (from scratch)
src/evaluate.py  held-out uplift report + JSON metrics
tests/test_smoke.py  domain smoke tests (CATE recovery, metric finiteness)
app.py           Streamlit dashboard
```

## Run
```sh
make install
make train     # trains metalearners, writes models/model.pkl + models/metrics.json
make test      # pytest -q
streamlit run app.py
```

## References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for
  estimating heterogeneous treatment effects*. PNAS.
- Radcliffe, N. J., & Surry, P. (2011). *Real uplift: Predictive modelling with
  classification trees in variable selection for uplift modelling* (Qini curve).
- Gutierrez, P., & Gérardy, J.-Y. (2017). *Causal inference and uplift modelling:
  a review of the literature*. PMLDC.
