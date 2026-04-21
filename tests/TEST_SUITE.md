# RSHAP — Test Suite Guide

The test suite is contained in `RSHAP_Test_Suite.ipynb`. It has three cells, each a self-contained suite with its own output. Run the cells top to bottom. All three suites are independent: each cell imports what it needs and uses its own data.

This package implements the method of:

> **Liu, T. & Barnard, A. S.** (2023). *Shapley Based Residual Decomposition for Instance Analysis.*
> ICML 2023, PMLR 202, pp. 21375–21387.
> [https://proceedings.mlr.press/v202/liu23b.html](https://proceedings.mlr.press/v202/liu23b.html)

Original research artefacts by **Dr Tommy Liu**: [github.com/uilymmot/residual-decomposition](https://github.com/uilymmot/residual-decomposition)

---

## Suite 1 — Unit Tests

**Cell label:** *Unit Test Suite*
**Framework:** `unittest`
**Tests:** 52
**Expected result:** 52 passed, 0 failures, 0 errors

This suite verifies the correctness of every public function and internal component in isolation, using small synthetic datasets (N=20, 3 features) so that each test runs in under a second.

### What is tested

#### `resolve_model`
- The `_MODEL_REGISTRY` contains all six expected keys (`ridge`, `logistic`, `svm`, `rf`, `mlp`, `xgb`)
- Each string name resolves to the correct sklearn/xgboost class for both regression and classification
- String matching is case-insensitive (`RF`, `SVM`, `XGB` all work)
- Passing a class object directly passes through unchanged
- `None` model params defaults to `{}`
- Explicit params are preserved without modification
- An unknown string raises `ValueError` with a message that lists valid names

#### `_compute_marginal_contributions`
- Returns a numpy ndarray of shape (N, N) and dtype float64
- Contains no NaN values for both regression and classification
- Classification values are integers (binarised residuals)
- Forward and reverse permutations produce different results (verifies symmetry is not trivially identity)
- A model that raises an exception causes the function to return `None` rather than propagating the crash
- Results are deterministic given the same permutation

#### `ResidualDecompositionSymmetric` attributes
- `phi` and `composition` attributes exist after `fit()`
- `n_jobs` is stored as provided
- `model_class` is stored as the resolved class object (not the string)
- `model_params` defaults to `{}`
- The `regression` flag is stored correctly

#### `ResidualDecompositionSymmetric` outputs
- `get_composition()` and `get_contribution()` return (N, N) arrays with no NaN values
- `get_composition()` is identical to `phi`
- `get_contribution()` satisfies the exact mathematical relationship: `contrib[i, j] = phi[i, j] × −sign(col_sum_i)`
- A class object (rather than a string) is accepted as `model_class`
- More iterations produce a different phi than fewer (phi is not static)
- `iterations=-1` (convergence mode) completes and returns a valid (N, N) matrix

#### Visualisation
- `draw_heatmap` returns a `(fig, ax)` tuple with non-None values
- `draw_heatmap` accepts an externally created axes object and draws into it
- `cc_plot` returns a scatter object for: no coloring, continuous coloring, categorical coloring, and group (`categorical_colouring`) mode

---

## Suite 2 — Algorithm Tests

**Cell label:** *Algorithm Test Suite*
**Tests:** 16
**Expected result:** 16/16 passed

This suite verifies that every supported model works end-to-end in both regression and classification modes, using slightly larger synthetic datasets (N=40, 5 features). It also checks string aliases and custom hyperparameter passing.

### What is tested

| Section | Tests |
|---|---|
| Regression | `ridge`, `svm`, `rf`, `mlp`, `xgb` — each produces a valid (N, N) phi matrix |
| Classification | Same five models in classification mode |
| String aliases | `'logistic'` (classification) and `'ridge'` (regression) work as aliases |
| Custom `model_params` | `Ridge(alpha=10)`, `SVR(C=0.5, rbf)`, `RF(10 trees, depth=3)`, `XGB(20 trees, depth=2)` — params are respected |

### Validity checks per model

Each model run is checked for:
- `get_composition()` shape == (N, N)
- `get_contribution()` shape == (N, N)
- No NaN values in either matrix
- Neither matrix is all-zero

Each result is printed with a ✓ or ✗ symbol and the wall-clock time. A summary line at the end shows the total pass count.

**Note on timing:** RF and MLP are slower than ridge or SVM because ensemble/neural-network fits are more expensive. On a typical laptop with `n_jobs=1`, RF and MLP may each take 15–30 seconds. This is expected.

---

## Suite 3 — Benchmark Tests

**Cell label:** *Benchmark Test Suite*
**Datasets:** 4 real scientific datasets downloaded at runtime
**Expected result:** all four benchmarks complete and print ✓ PASS

This suite demonstrates a distinct RSHAP capability on each of four public scientific datasets. Unlike the first two suites, each benchmark is designed to show that RSHAP produces meaningful, interpretable results — not just that it runs without error.

### Data loading

Benchmarks 1, 3, and 4 are loaded from OpenML via `sklearn.datasets.fetch_openml`. Benchmark 2 is downloaded directly from the UCI Machine Learning Repository. An internet connection is required on first run; OpenML results are cached locally by sklearn after that.

A `load_openml()` helper handles the case where some OpenML datasets return `target=None` (the target column is in `ds.frame` rather than `ds.target`). This is handled transparently.

### Benchmark 1 — Concrete Compressive Strength

**Model:** Ridge regression
**Dataset:** 1,030 concrete mix designs with compressive strength (MPa) as target (OpenML data_id=4353)
**Capability demonstrated:** Influential instance identification

RSHAP's composition score (column sum of phi) measures how much each concrete mix shifts the prediction errors of every other mix in the dataset. The benchmark:
- Prints the five most positively influential mixes (those that consistently drive errors up for others)
- Prints the five most negatively influential mixes (those that suppress errors for others)
- Reports which ingredient feature is most correlated with the composition score
- Passes if the composition scores have non-zero variance (i.e., some mixes are genuinely more influential than others)

### Benchmark 2 — NASA Airfoil Self-Noise

**Model:** Random Forest regression
**Dataset:** 1,503 aerodynamic measurements (frequency, angle, chord length, velocity, suction side displacement thickness) with measured sound pressure in dB as target (UCI archive)
**Capability demonstrated:** CC plot with continuous coloring

The CC plot is produced with points colored by measured dB level. The benchmark then quantifies whether aerodynamically extreme aerofoils (top and bottom 10% of dB readings) appear as outliers in RSHAP space compared to central aerofoils, by computing the mean Euclidean distance from the origin in CC coordinates. Passes if extreme-dB instances are more displaced than central ones.

### Benchmark 3 — QSAR Biodegradation

**Model:** SVM classification
**Dataset:** 1,055 molecular compounds (41 molecular descriptors) labeled as biodegradable or not (OpenML data_id=1494)
**Capability demonstrated:** CC plot class separation

The CC plot is produced with points colored by class. The benchmark reports the centroid of each class in CC space and the Euclidean distance between centroids. Class separation in RSHAP space indicates that the model's residual structure is systematically different for biodegradable vs. non-biodegradable compounds. Passes if the centroids are not coincident.

### Benchmark 4 — Wine Quality Red

**Model:** XGBoost classification
**Dataset:** 1,599 red wine samples (11 physicochemical features) with quality scores binarised into high (≥6) vs. low (<6) quality (OpenML data_id=40691)
**Capability demonstrated:** Inter-class contribution heatmap

`draw_heatmap` is called with high/low quality labels to produce a 2×2 inter-class contribution grid. The benchmark then reports the mean contribution from high-quality instances to low-quality ones and vice versa, and computes their ratio. Asymmetry between these two cells shows that the two quality tiers exert different levels of influence on each other's predictions. Passes if the two cross-class means are not identical.

---

## Running the suites

Open `RSHAP_Test_Suite.ipynb` in Jupyter and run all cells in order. Each suite is self-contained and can also be re-run independently.

To run Suite 1 as a standalone script (outside Jupyter):

```bash
python run_unit_tests.py
```

---

## Interpreting failures

| Failure | Likely cause |
|---|---|
| `ImportError` on any model | Missing dependency — check `requirements.txt` and reinstall |
| `BrokenProcessPool` / `[WinError 1455]` | Windows paging file too small — set `N_JOBS = 1` at the top of the suite cell |
| Benchmark dataset not found | Internet connection required; check firewall or proxy settings |
| `AttributeError: 'NoneType'.values` | OpenML returned `target=None` — the `load_openml()` helper handles this; ensure you have the latest version of `RSHAP_Test_Suite.ipynb` |
| `test_more_iterations_changes_phi` fails | Random seed collision (extremely unlikely) — re-run the cell |
| Benchmark ✗ FAIL on shape/NaN | Model crashed silently — check `rshap_error.log` in the working directory |
