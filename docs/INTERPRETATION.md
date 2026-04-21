# RSHAP — Interpreting the Results

This document explains what the phi matrix, composition scores, and contribution scores mean in practice, and how to reason about the results RSHAP produces.

The theoretical foundations of these measures are established in:

> **Liu, T. & Barnard, A. S.** (2023). *Shapley Based Residual Decomposition for Instance Analysis.*
> ICML 2023, PMLR 202, pp. 21375–21387.
> [https://proceedings.mlr.press/v202/liu23b.html](https://proceedings.mlr.press/v202/liu23b.html)

---

## The phi matrix

After calling `rshap.fit(...)`, the central output is `phi` — an N×N matrix accessible via `rshap.get_composition()`.

```
phi[k, j] = average change in instance j's prediction residual
             when instance k is added to the training coalition
```

Think of it as a directed influence table: **row k describes what instance k does to everyone else; column j describes what everyone does to instance j.**

- `phi[k, j] > 0`: adding k to the training set tends to *increase* j's prediction error
- `phi[k, j] < 0`: adding k to the training set tends to *decrease* j's prediction error (better prediction of j)
- `phi[k, j] ≈ 0`: instance k has little or no effect on instance j's residual

The diagonal `phi[k, k]` is meaningful: it captures how much including instance k in its own training coalition changes the model's ability to predict k itself.

---

## Composition score — how much does each instance affect others?

```python
composition_score = rshap.get_composition().sum(axis=0)  # shape: (N,)
```

The **composition score** of instance j is the column sum of phi — the total change in j's residual accumulated across all addition events and all permutations.

| Score | Meaning |
|---|---|
| Large positive | j's residual is consistently driven **up** as the training set grows around it — j is systematically difficult for the model to predict regardless of what data is available |
| Large negative | j's residual is consistently **suppressed** as the training set grows — j is easy to predict, and its error is reliably reduced by adding more data |
| Near zero | j behaves like an average instance — neither particularly hard nor easy to predict |

**In practice**, instances with extreme composition scores are worth examining individually. High-positive instances are often:
- Genuine anomalies or outliers
- Instances at the boundary between distinct data regimes
- Mislabelled data points

High-negative instances tend to be:
- Highly typical, well-represented examples
- Instances that anchor the model's understanding of a region of feature space

---

## Contribution score — how does each instance affect others?

```python
contribution = rshap.get_contribution()    # N×N matrix
contribution_score = contribution.sum(axis=1)  # shape: (N,)
```

`get_contribution()` returns a sign-adjusted version of phi. Each row i of phi is multiplied by `−sign(composition_score_i)`:

```
contribution[k, j] = phi[k, j] × (−sign(composition_score_k))
```

This sign flip has a specific purpose: it aligns the sign convention so that positive contribution values consistently mean *helpful* influence — instance k's entry into the training coalition improves predictions — regardless of the direction of the raw residual change.

The **contribution score** (row sum of the contribution matrix) summarises how strongly each instance influences the prediction accuracy of all others:

- **High positive contribution score**: instance k reliably improves the model for others when included in training
- **High negative contribution score**: instance k degrades others' predictions when included

---

## What the CC plot axes mean

The CC (Composition–Contribution) plot places every instance at coordinates:

```
x = composition score (column sum of phi)
y = contribution score (row sum of contribution matrix)
```

The two axes measure different things:

| Axis | What it measures |
|---|---|
| Composition (x) | How much others' additions affect *this* instance's own residual |
| Contribution (y) | How much *this* instance's addition affects everyone else's residuals |

The four quadrants have natural interpretations:

```
                     Contribution (+)
                  (helps others when added)
                            │
  Composition (−)           │           Composition (+)
  (own error                │            (own error
   reliably reduced)        │             stays high)
  ─────────────────────────────────────────────────────
                            │
  Composition (−)           │           Composition (+)
  (own error                │            (own error
   reliably reduced)        │             stays high)
                            │
                     Contribution (−)
                  (hurts others when added)
```

**Top-left quadrant** (Composition −, Contribution +): these instances are easy to predict themselves *and* benefit others — typical, representative examples.

**Top-right quadrant** (Composition +, Contribution +): hard to predict themselves, but their inclusion still helps others — potentially informative boundary cases.

**Bottom-left quadrant** (Composition −, Contribution −): easy to predict but they degrade others' predictions — redundant or misleading instances.

**Bottom-right quadrant** (Composition +, Contribution −): hard to predict *and* harmful to others — strong candidates for anomaly or noise investigation.

---

## Group-level interpretation — the heatmap

`draw_heatmap(rshap, labels)` aggregates the contribution matrix into a group×group grid. Each cell `(i, j)` shows the mean contribution value from instances in group i to instances in group j.

The values are normalised to the range `[−1, +1]` separately for positive and negative entries, so the colormap always spans the full range regardless of absolute magnitude.

**Rows** represent the "giving" group — instances whose marginal addition is being measured.
**Columns** represent the "receiving" group — instances whose residuals are being changed.

A strongly positive cell `(i, j)` means that instances from group i consistently help the model predict instances from group j. A strongly negative cell means the opposite.

In a well-separated classification problem, diagonal cells (within-class influence) tend to be positive, and off-diagonal cells (cross-class influence) tend to be negative, because adding more examples of class A helps predict other class A instances but may confuse the model about class B.

---

## Regression vs. classification

**Regression**: phi entries are continuous and in the units of the residual (i.e., the same units as the target variable). A value of `phi[k, j] = 2.5` means that adding instance k shifted j's prediction error by an average of 2.5 units across all permutations where k was the marginal addition.

**Classification**: residuals are binarised to 0 (correct prediction) or 1 (incorrect prediction) before accumulation. The phi matrix therefore contains values in the range `[−1, +1]` and represents changes in *misclassification rates* rather than continuous errors. A value of `phi[k, j] = 0.3` means that adding instance k increases the probability of misclassifying j by about 30 percentage points on average.

---

## Practical considerations

**More iterations → smoother estimates.** The phi matrix is an average over random permutations. With few iterations the estimates are noisy, especially for large N. For exploratory work, 20–50 iterations on a fast model is sufficient to identify coarse structure. For publication or final analysis, 100–200+ iterations is recommended.

**Model choice affects what RSHAP measures.** RSHAP is not model-agnostic in the way that SHAP is — the phi values reflect residuals *of the chosen model*, not some ground-truth influence. A ridge regression and a random forest will produce different phi matrices on the same data, because the two models have different inductive biases and therefore different residual patterns. Choose a model that is appropriate for the task and data type.

**Convergence mode is useful for unfamiliar datasets.** If you are unsure how many iterations are sufficient, use `iterations=-1` to let RSHAP run until the phi matrix stabilises. The convergence threshold of 2.5% mean relative change is a conservative stopping rule.

**Small datasets converge faster.** The variance of the Shapley estimator scales roughly as N! / (number of permutations), so smaller datasets reach stable estimates with fewer iterations than larger ones.
