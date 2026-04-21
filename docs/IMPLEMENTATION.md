# RSHAP — Implementation Notes

This document describes how RSHAP is implemented internally. It is intended for contributors, researchers who want to understand the algorithm, and users who need to reason about computational cost or numerical behaviour.

The method is described in full in:

> **Liu, T. & Barnard, A. S.** (2023). *Shapley Based Residual Decomposition for Instance Analysis.*
> ICML 2023, PMLR 202, pp. 21375–21387.
> [https://proceedings.mlr.press/v202/liu23b.html](https://proceedings.mlr.press/v202/liu23b.html)

The original research artefacts are maintained by **Dr Tommy Liu** at
[github.com/uilymmot/residual-decomposition](https://github.com/uilymmot/residual-decomposition).

---

## Overview

RSHAP estimates a Shapley-style decomposition of model residuals across training instances. The core idea is to ask, for every pair of instances (k, j): *by how much does instance k's marginal entry into the training set change the prediction error at instance j?*

The answer is captured in the **phi matrix** — an N×N array where N is the number of training instances. Each entry `phi[k, j]` is an averaged marginal residual change, estimated over many random orderings of the data.

---

## User workflow and how it maps to the implementation

### What RSHAP expects — and what it does not

In a typical workflow you will have your own trained model used for making predictions. RSHAP is a **separate, independent analysis** — it does not receive your trained model as input and does not extend it.

```python
# Your model — trained and used for prediction as normal
model = XGBClassifier(**xgb_params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# RSHAP — run independently on the same data
# Pass the CLASS and its params, not the fitted model instance above
rshap = ResidualDecompositionSymmetric()
rshap.fit(X_train, y_train,
          model_class=XGBClassifier,  # the class, not the fitted 'model' above
          model_params=xgb_params,
          iterations=100, regression=False)
```

`model_class` and `model_params` tell RSHAP how to **construct** the model — not a pre-fitted instance. RSHAP instantiates a fresh copy of the model internally for every permutation evaluation, trains it from scratch on the current coalition, and discards it. Your own trained model is never involved.

**`model_params` must match the hyperparameters used to train your prediction model.** RSHAP's residual decomposition is only meaningful if the model it trains internally behaves the same way as the model you are analysing. If you use different hyperparameters, the phi matrix will reflect a different model's residuals — not the one you actually deployed — and the results will not be interpretable as an analysis of your model.

### The three-step workflow

```python
# Step 1 — provide raw data and specify the model type
#   model_class and model_params describe how to BUILD the model,
#   not a pre-trained model instance.
rshap = ResidualDecompositionSymmetric()
rshap.fit(X, y,
          model_class='ridge',         # string shorthand, or pass Ridge directly
          model_params={'alpha': 1.0}, # constructor arguments for that model
          iterations=50,               # number of symmetric permutation pairs
          regression=True,             # True = continuous target, False = labels
          n_jobs=1)                     # parallel workers

# Step 2 — retrieve the phi matrix and its signed variant
phi     = rshap.get_composition()   # returns self.phi directly — no extra computation
contrib = rshap.get_contribution()  # lightweight: sign-adjusted view of phi

# Step 3 — visualise
rshap.cc_plot(coloring=y)           # draws on the current matplotlib figure
draw_heatmap(rshap, labels)         # returns (fig, ax)
```

**`fit()` must be called before any other method.** Calling `get_composition()`, `get_contribution()`, `cc_plot()`, or `draw_heatmap()` on an unfitted object will raise an `AttributeError` because `self.phi` does not yet exist.

### What `fit()` does internally

1. **`resolve_model()`** — converts the `model_class` string to the correct sklearn or XGBoost class (e.g. `'ridge'` + `regression=True` → `Ridge`; `'ridge'` + `regression=False` → `LogisticRegression`). If a class is passed directly it is used as-is. `model_params` defaults to `{}` if `None`.
2. **`iteration_fit()` or `convergence_fit()`** — generates symmetric permutation pairs and dispatches each to a worker via `joblib.Parallel`. Each worker independently instantiates the model using `model_class(**model_params)`, runs the full coalition sequence, and returns a partial phi matrix.
3. **Accumulation and normalisation** — the partial matrices from all workers are summed and divided by the number of permutations, producing `self.phi`.

`get_composition()` and `get_contribution()` are both lightweight accessors — all the computation happens inside `fit()`.

---

## The marginal contribution function

For a given permutation `σ` of the N training indices, the algorithm builds the training coalition incrementally:

1. Start with the singleton `{σ[0]}`
2. At each step i, add `σ[i]` to the current coalition
3. Fit the model on the current coalition and predict on the full dataset
4. Record the change in the residual vector compared to the previous step
5. Assign that change to row `σ[i]` of the phi matrix

Formally, let `R(S)` be the residual vector when the model is trained on subset S and evaluated on all N instances. The marginal contribution of instance k at position i in permutation σ is:

```
Δ(k, σ) = R(σ[:i+1]) − R(σ[:i])
```

This is added to row k of phi. After averaging over all permutations, `phi[k, j]` is the expected change in instance j's residual attributable to instance k's entry into the training coalition.

---

## Symmetric permutation pairs

A key design choice is that every random permutation `p` is paired with its reverse `p[::-1]`. For every pair:

- Forward pass: instances are added in order `p[0], p[1], ..., p[N-1]`
- Reverse pass: instances are added in order `p[N-1], ..., p[1], p[0]`

This symmetry guarantees that every instance appears at every coalition position an equal number of times across a pair, which substantially reduces variance and means fewer total permutations are needed to reach a stable estimate.

The `iterations` parameter controls the **number of pairs**. The total number of permutations evaluated is therefore `2 × iterations`.

---

## Regression vs. classification residuals

**Regression** (`regression=True`):

The value function is:
```python
model.fit(X_coalition, y_coalition)
residuals = model.predict(X_full) − y_full
```
Residuals are continuous real values.

**Classification** (`regression=False`):

The residual is binarised: after fitting and predicting, the difference between predicted and true label is mapped to `{0, 1}` — zero if the prediction is correct, one if it is wrong. This makes the phi matrix track *misclassification events* rather than continuous error magnitudes.

Edge cases handled:
- If the coalition contains only one class, the model cannot be trained normally; the algorithm fills residuals with a constant based on the single available class.
- If the coalition's class set is a strict subset of the full label set, labels are re-encoded before fitting to ensure the model does not see unknown classes at training time.

---

## `iteration_fit` — fixed permutation count

```python
rshap.fit(X, y, model_class='ridge', iterations=50, n_jobs=1)
```

Steps:
1. Generate `iterations // 2` random permutations and append each reverse
2. Dispatch all permutations to `joblib.Parallel` with the `loky` backend
3. Accumulate the resulting phi matrices
4. Divide by `iterations` to normalise

The final phi represents the average marginal residual change per permutation pair.

**Choosing `iterations`**: for small datasets (N < 50), 20–50 iterations is usually sufficient. For larger datasets, 100–200 may be needed. Use the convergence mode if unsure.

---

## `convergence_fit` — automatic stopping

```python
rshap.fit(X, y, model_class='ridge', iterations=-1, n_jobs=1)
```

Instead of a fixed count, the algorithm runs in blocks of N permutation pairs and checks whether the phi matrix has stabilised:

1. Run a block of N symmetric pairs
2. Compute the running average phi
3. Compare to the previous block's average: measure the mean absolute relative change across all non-zero entries
4. Print progress every 10 blocks
5. Stop when the mean change falls below **2.5%** (threshold `0.025`)
6. Cap at 200 blocks to prevent infinite loops

Convergence mode is more reliable for unusual datasets but is slower because it cannot be fully pre-batched for parallelism.

---

## Parallelism

All permutations are evaluated in parallel using `joblib.Parallel` with the `loky` process-based backend. Each worker calls `_compute_marginal_contributions` independently — there is no shared state.

**`n_jobs` parameter:**
- `-1` — use all available CPU cores (default)
- `1` — sequential execution (no subprocesses)
- `k` — use exactly k cores

**Windows paging file issue**: The `loky` backend spawns subprocesses that each re-import all dependencies (numpy, pandas, sklearn, xgboost). On Windows systems with limited virtual memory, this can trigger `[WinError 1455] The paging file is too small`. Use `n_jobs=1` to run sequentially without subprocess overhead.

**Error handling**: If a worker crashes (e.g., due to a model that throws an exception), `_compute_marginal_contributions` catches the exception, logs it to `rshap_error.log`, and returns `None`. The main process skips `None` results with a warning rather than crashing. This means a run with occasional worker failures still produces a valid (if noisier) phi estimate.

---

## Computational complexity

Each permutation evaluation requires N−1 sequential model fits (one per coalition step) and N predictions per fit. Total fits per run: `2 × iterations × (N − 1)`.

| Dataset size | Iterations | Total fits |
|---|---|---|
| N = 50 | 50 | 4,900 |
| N = 100 | 100 | 19,800 |
| N = 500 | 50 | 49,500 |

Model fit time dominates. Fast models (ridge, linear SVM) complete in seconds; ensemble models (RF, XGB) or neural networks (MLP) can take minutes to hours on larger datasets.

**Practical guidance:**
- Start with a fast model (`ridge`) to check that results are sensible before switching to a slower one
- Use `n_jobs=-1` when memory allows; fall back to `n_jobs=1` on constrained machines
- For exploratory work, `iterations=10–20` with a fast model gives a useful rough picture

---

## Module structure

```
RSHAP.py
├── _MODEL_REGISTRY              # string → (regression class, classification class)
├── resolve_model()              # string/class resolver
├── __value_function()           # regression residual computation
├── __val_func_classification()  # classification residual computation
├── _compute_marginal_contributions()  # single permutation worker
├── __get_heatmap()              # heatmap aggregation (internal)
├── draw_heatmap()               # public heatmap visualisation
└── ResidualDecompositionSymmetric
    ├── fit()
    ├── iteration_fit()
    ├── convergence_fit()
    ├── get_composition()
    ├── get_contribution()
    └── cc_plot()
```
