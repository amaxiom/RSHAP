# RSHAP — Residual Decomposition Symmetric

RSHAP is a Python package for **instance-level influence analysis** of machine learning models. It uses a Shapley-style residual decomposition to measure how much each training instance affects the prediction errors of every other instance, producing a rich picture of inter-instance relationships that standard feature-importance methods cannot provide.

> This package implements and extends the method introduced in:
>
> **Liu, T. & Barnard, A. S.** (2023). *Shapley Based Residual Decomposition for Instance Analysis.*
> Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR 202, pp. 21375–21387.
> [https://proceedings.mlr.press/v202/liu23b.html](https://proceedings.mlr.press/v202/liu23b.html)
>
> The original research artefacts and paper notebooks are maintained by **Dr Tommy Liu** at
> [github.com/uilymmot/residual-decomposition](https://github.com/uilymmot/residual-decomposition).

---

## What RSHAP does

Where classical SHAP answers *"which features matter for this prediction?"*, RSHAP answers *"which training instances matter, and for whom?"*

For a dataset of N instances, RSHAP produces an **N×N composition matrix** (`phi`). Each entry `phi[k, j]` captures the marginal change in instance j's prediction residual when instance k is added to the training coalition. Aggregating this matrix in different ways reveals:

- **Which instances are the most influential** — driving up or suppressing errors across the whole dataset
- **Which instances are the most sensitive** — whose predictions are most affected by others
- **Inter-group dynamics** — whether instances from one class or category systematically help or hinder predictions for another

---

## Installation

```bash
pip install -r requirements.txt
```

Or install the dependencies directly:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib tqdm joblib
```

Clone this repo and import `RSHAP` directly — no package installation step is required for local use:

```python
import sys
sys.path.insert(0, "/path/to/RSHAP")
from RSHAP import ResidualDecompositionSymmetric
```

---

## How RSHAP works — and what you need to provide

In a typical machine learning workflow you train a model to make predictions. RSHAP is a **separate analysis** that runs alongside that — it is not an extension of your trained model and does not receive it as input.

You will usually have two independent steps:

```python
# Step A — your own model, trained for prediction and evaluation as normal
model = XGBClassifier(**xgb_params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)   # use this model for your results

# Step B — RSHAP analysis, run independently on the same data
#   Pass the model CLASS and its params — not the trained model instance above.
#   RSHAP creates its own fresh instances of the model internally.
rshap = ResidualDecompositionSymmetric()
rshap.fit(X_train, y_train,
          model_class=XGBClassifier,   # the class, not the fitted 'model' above
          model_params=xgb_params,
          iterations=100, regression=False)
```

`model` and `rshap` are completely independent. RSHAP never sees your trained model. It uses the class and parameters you provide to build and train hundreds of its own fresh model instances internally, each on a different random subset of the data, in order to measure how each training instance affects the prediction errors of every other.

What you provide to `rshap.fit()`:
- `X`, `y` — your raw data (same data you trained your model on)
- `model_class` — the **type** of model to use, as a string (`'xgb'`, `'ridge'`, `'svm'`, `'rf'`, `'mlp'`) or a class directly (`XGBClassifier`, `Ridge`, etc.)
- `model_params` — **must be the same hyperparameters you used to train your own model.** RSHAP's results reflect the residuals of the model it trains internally — if the hyperparameters differ, the phi matrix will describe a different model's behaviour, not the one you are analysing.

What RSHAP does with these:
- Generates many random orderings of your N instances
- For each ordering, trains a fresh instance of the specified model on progressively larger subsets, recording how each instance's addition changes the prediction errors of all others
- Averages the results into the N×N phi matrix

---

## Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from RSHAP import ResidualDecompositionSymmetric, draw_heatmap

# Step 1 — prepare your data (no model training needed)
X = np.random.randn(100, 5)
y = X @ [1.5, -1.0, 0.5, 0.2, -0.8] + np.random.randn(100) * 0.3

# Step 2 — tell RSHAP which model type to use and run the analysis
#   model_class  : string name or class — RSHAP instantiates and trains this internally
#   model_params : hyperparameters for that model (optional, defaults to {})
#   iterations   : how many permutation pairs to average over
#   regression   : True for continuous targets, False for classification labels
#   n_jobs       : parallel workers (-1 = all cores, 1 = sequential / safer on Windows)
rshap = ResidualDecompositionSymmetric()
rshap.fit(X, y, model_class='ridge', model_params={'alpha': 1.0},
          iterations=50, regression=True, n_jobs=1)

# Step 3 — retrieve the N×N phi matrix and its signed variant
phi     = rshap.get_composition()   # raw phi matrix
contrib = rshap.get_contribution()  # sign-adjusted matrix

influence = phi.sum(axis=0)         # per-instance influence score (column sums)
print("Most influential instance:", influence.argmax())

# Step 4 — visualise
plt.figure(figsize=(7, 5))
rshap.cc_plot(coloring=y)
plt.title("CC Plot")
plt.show()

labels = np.array(["GroupA"] * 50 + ["GroupB"] * 50)
fig, ax = draw_heatmap(rshap, labels)
plt.show()
```

`fit()` must be called before retrieving results or producing visualisations. Once it has completed, the fitted object can be passed to `draw_heatmap` or used to call `cc_plot` as many times as needed without re-running the analysis.

### Classification

The workflow is identical — set `regression=False` and name your preferred model. RSHAP automatically selects the classification variant (e.g. `SVR` → `SVC`, `Ridge` → `LogisticRegression`):

```python
rshap = ResidualDecompositionSymmetric()
rshap.fit(X, y_binary, model_class='svm', model_params={'C': 1.0},
          iterations=50, regression=False, n_jobs=1)
```

### Passing a model class directly

If you prefer not to use the string shorthand, pass the class and its parameters explicitly:

```python
from sklearn.ensemble import RandomForestRegressor

rshap = ResidualDecompositionSymmetric()
rshap.fit(X, y, model_class=RandomForestRegressor,
          model_params={'n_estimators': 100, 'max_depth': 5},
          iterations=50, regression=True, n_jobs=1)
```

---

## Supported models

Pass a model as a **string name** or as a **class object**. The string interface automatically selects the regression or classification variant based on the `regression` flag.

| String | Regression class | Classification class |
|---|---|---|
| `'ridge'` | `Ridge` | `LogisticRegression` |
| `'logistic'` | `Ridge` | `LogisticRegression` |
| `'svm'` | `SVR` | `SVC` |
| `'rf'` | `RandomForestRegressor` | `RandomForestClassifier` |
| `'mlp'` | `MLPRegressor` | `MLPClassifier` |
| `'xgb'` | `XGBRegressor` | `XGBClassifier` |

**Custom class:**

```python
from sklearn.linear_model import Ridge
rshap.fit(X, y, model_class=Ridge, model_params={'alpha': 2.0}, iterations=50)
```

**Custom hyperparameters:**

```python
rshap.fit(X, y, model_class='xgb',
          model_params={'n_estimators': 200, 'max_depth': 4},
          iterations=100, regression=True, n_jobs=-1)
```

---

## API reference

### `ResidualDecompositionSymmetric`

#### `fit(X, y, model_class, model_params, iterations, regression, n_jobs)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array (N, F) | — | Feature matrix |
| `y` | array (N,) | — | Target vector |
| `model_class` | str or class | `None` | Model to use (see table above) |
| `model_params` | dict | `None` | Keyword arguments passed to the model constructor |
| `iterations` | int | `100` | Number of symmetric permutation pairs. Pass `-1` for automatic convergence. |
| `regression` | bool | `True` | `True` for regression, `False` for classification |
| `n_jobs` | int | `-1` | Parallel workers (`-1` = all cores, `1` = sequential) |

#### `get_composition()` → ndarray (N, N)
Returns the raw phi matrix.

#### `get_contribution()` → ndarray (N, N)
Returns a sign-adjusted version of phi. Each row i is scaled by `−sign(column_sum_i)`, so that positive values consistently indicate a helpful influence on prediction accuracy.

#### `cc_plot(coloring, fontsizes, axis_lines, cc_function, categorical_colouring, drawparams, legend_label)` → scatter object
Plots the CC plot on the current matplotlib axes. See [VISUALISATION_GUIDE.md](VISUALISATION_GUIDE.md).

### `draw_heatmap(rshap_object, labels, decimals, num_ticks, fontsizes, ax)` → (fig, ax)
Plots an inter-group contribution heatmap. See [VISUALISATION_GUIDE.md](VISUALISATION_GUIDE.md).

---

## Parallelism and memory

RSHAP uses `joblib` to parallelise permutation evaluations. On Windows, spawning many subprocesses that each import heavy libraries (pandas, sklearn, xgboost) can exhaust the virtual memory paging file. If you encounter `[WinError 1455]` or `BrokenProcessPool` errors, set `n_jobs=1`:

```python
rshap.fit(X, y, model_class='rf', iterations=50, n_jobs=1)
```

---

## Convergence mode

Pass `iterations=-1` to run until the phi matrix stabilises rather than for a fixed number of iterations:

```python
rshap.fit(X, y, model_class='ridge', iterations=-1, n_jobs=1)
```

The algorithm checks convergence every 10 iteration blocks and stops when the mean absolute relative change across the phi matrix falls below 2.5%.

---

## Documentation

| Document | Contents |
|---|---|
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Algorithm internals — permutations, phi matrix, convergence |
| [INTERPRETATION.md](INTERPRETATION.md) | What the results mean and how to read them |
| [VISUALISATION_GUIDE.md](VISUALISATION_GUIDE.md) | CC plots and heatmaps explained |
| [TEST_SUITE.md](TEST_SUITE.md) | Guide to the test suite notebook |

---

## Citation

If you use RSHAP in your research, please cite the original paper:

```bibtex
@InProceedings{pmlr-v202-liu23b,
  title     = {Shapley Based Residual Decomposition for Instance Analysis},
  author    = {Liu, Tommy and Barnard, Amanda S},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages     = {21375--21387},
  year      = {2023},
  volume    = {202},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

The original research artefacts accompanying the paper are maintained by **Dr Tommy Liu** at
[github.com/uilymmot/residual-decomposition](https://github.com/uilymmot/residual-decomposition).

---

## Authors

- **Prof Amanda S Barnard** — [github.com/amaxiom](https://github.com/amaxiom)
- **Dr Tommy Liu** — [github.com/uilymmot](https://github.com/uilymmot)

---

## License

MIT — see [LICENSE](LICENSE).
