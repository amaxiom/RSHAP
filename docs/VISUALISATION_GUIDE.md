# RSHAP — Visualisation Guide

RSHAP provides two visualisations: the **CC plot** (`cc_plot`) and the **contribution heatmap** (`draw_heatmap`). This guide covers every available option for both, with examples and notes on how to read the output.

The CC plot and heatmap are introduced in:

> **Liu, T. & Barnard, A. S.** (2023). *Shapley Based Residual Decomposition for Instance Analysis.*
> ICML 2023, PMLR 202, pp. 21375–21387.
> [https://proceedings.mlr.press/v202/liu23b.html](https://proceedings.mlr.press/v202/liu23b.html)

---

## Before you visualise — fitting RSHAP

Both visualisations require a fitted `ResidualDecompositionSymmetric` object. RSHAP handles all model training internally: you provide your data and choose a model type, and RSHAP repeatedly fits that model across random permutations of the training set to build the phi matrix. You do not need to pre-fit a model yourself.

The full workflow is:

```python
import numpy as np
import matplotlib.pyplot as plt
from RSHAP import ResidualDecompositionSymmetric, draw_heatmap

# 1. Prepare your data
X = ...   # array of shape (N, n_features)
y = ...   # array of shape (N,)

# 2. Create and fit the RSHAP explainer
#    - model_class: the model RSHAP will use internally (string or class)
#    - iterations:  number of symmetric permutation pairs to average over
#    - regression:  True for a continuous target, False for classification
#    - n_jobs:      parallel workers (-1 = all cores, 1 = sequential)
rshap = ResidualDecompositionSymmetric()
rshap.fit(X, y, model_class='ridge', iterations=50, regression=True, n_jobs=1)

# 3. Visualise
plt.figure(figsize=(8, 6))
rshap.cc_plot()
plt.show()
```

**Model choice matters.** The phi matrix reflects the residuals of the model you choose, so different models will produce different visualisations on the same data. Start with `'ridge'` for a fast, linear baseline. Switch to `'rf'`, `'svm'`, `'mlp'`, or `'xgb'` if you need a model that better captures the structure of your data.

**Iterations and runtime.** Each iteration fits the model N−1 times (once per coalition step) for both a forward and reverse permutation. More iterations give a smoother, lower-variance phi matrix but take proportionally longer. For exploratory visualisation, 20–50 iterations with `'ridge'` is usually sufficient. For final analysis or slower models, use more.

**`n_jobs=1` on Windows.** If you encounter a `BrokenProcessPool` error, set `n_jobs=1` to disable multiprocessing. See the README for details.

After `fit()` completes, `rshap.get_composition()` and `rshap.get_contribution()` are available and all visualisation functions can be called.

---

## The CC plot

### What it shows

The CC plot places every training instance at a point in 2D space:

- **x-axis — Composition sum**: the column sum of the phi matrix for that instance. Measures how much this instance's own prediction residual changes as other instances are progressively added to the training set.
- **y-axis — Contribution sum**: the row sum of the contribution matrix. Measures how much this instance's own entry into the training set shifts the prediction residuals of all other instances.

Together, the two axes capture both sides of inter-instance influence: how an instance is *affected* by the rest of the data (x), and how it *affects* the rest of the data (y). Instances that behave similarly cluster together; anomalous or highly influential instances appear displaced from the main cloud.

For a full explanation of the quadrant meanings see [INTERPRETATION.md](INTERPRETATION.md).

---

### Basic usage

`cc_plot` draws onto the **currently active matplotlib figure**. Always create a figure first:

```python
import matplotlib.pyplot as plt
from RSHAP import ResidualDecompositionSymmetric

# ... fit rshap ...

plt.figure(figsize=(8, 6))
rshap.cc_plot()
plt.title("CC Plot")
plt.show()
```

When called with no arguments, points are colored by the training target `y` using a continuous viridis colormap.

---

### Complete parameter reference

```python
rshap.cc_plot(
    coloring            = None,
    fontsizes           = 16,
    axis_lines          = True,
    cc_function         = np.sum,
    categorical_colouring = None,
    drawparams          = {'marker': 'o'},
    legend_label        = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `coloring` | array-like or `None` | `None` | Values used to color each point. If `None`, defaults to the training target `y`. See [Coloring modes](#coloring-modes) below. Ignored when `categorical_colouring` is set. |
| `fontsizes` | int | `16` | Font size for axis labels and legend text. The axis labels in group mode are additionally scaled by 1.2×. |
| `axis_lines` | bool | `True` | If `True`, draws a horizontal orange line at y=0 and a vertical red line at x=0, dividing the plot into four quadrants. |
| `cc_function` | callable | `np.sum` | Aggregation function applied along the composition and contribution matrices to produce the x and y coordinates. Must accept an `axis` keyword argument (e.g. `np.sum`, `np.mean`, `np.max`). |
| `categorical_colouring` | array-like or `None` | `None` | When set, switches to **group mode** (see below). Each group gets its own scatter series aggregated separately. Takes priority over `coloring`. |
| `drawparams` | dict | `{'marker': 'o'}` | Any additional keyword arguments passed directly to `plt.scatter` — e.g. `s` (marker size), `alpha`, `edgecolors`, `linewidths`. |
| `legend_label` | str or `None` | `None` | Title string for the legend. Only displayed when a legend is produced (categorical or group coloring). If `None`, the legend has no title. |

**Returns:** the scatter object (or list of scatter objects in group mode), for downstream use with `plt.colorbar` or similar.

---

### Coloring modes

All coloring uses the **viridis** colormap throughout.

#### Default — continuous target coloring

```python
plt.figure(figsize=(8, 6))
rshap.cc_plot()
plt.show()
```

Points are colored by the training target `y` using a continuous viridis scale. No colorbar is added automatically; add one manually if needed:

```python
sc = rshap.cc_plot()
plt.colorbar(sc, label="Target value")
```

#### Continuous coloring — any scalar array

```python
plt.figure(figsize=(8, 6))
sc = rshap.cc_plot(coloring=y)
plt.colorbar(sc, label="Compressive strength (MPa)")
plt.show()
```

Pass any float array of length N. Values are mapped continuously through viridis. Useful for:
- Target variable: do high/low values cluster in RSHAP space?
- A single feature: does one feature drive the influence structure?
- Prediction error: which instances does the model consistently get wrong?

#### Discrete / categorical coloring — class labels or integer codes

```python
labels = np.array(["Biodegradable"] * 700 + ["Non-biodegradable"] * 355)

plt.figure(figsize=(9, 6))
rshap.cc_plot(coloring=labels, legend_label="Class")
plt.show()
```

Pass an array of strings or integers. RSHAP detects the dtype and assigns each unique value a distinct viridis color with a legend entry. This answers: *do different classes occupy different regions of RSHAP space?*

The legend is positioned outside the right edge of the axes. Use `bbox_inches='tight'` when saving to avoid clipping it:

```python
fig, ax = plt.subplots(figsize=(9, 6))
rshap.cc_plot(coloring=labels, legend_label="Class")
fig.savefig("cc_plot.png", dpi=150, bbox_inches='tight')
```

#### Group mode — `categorical_colouring`

```python
plt.figure(figsize=(9, 6))
rshap.cc_plot(categorical_colouring=labels, legend_label="Group")
plt.show()
```

This is a fundamentally different mode from `coloring`. Instead of coloring individual instances, RSHAP computes separate composition and contribution scores *for each group*, restricted to that group's own rows and columns of the phi matrix:

- **x-axis per group**: composition of the group's instances (their columns of phi)
- **y-axis per group**: contribution of the group's instances (their rows of the contribution matrix)

Each group produces its own scatter cloud. This reveals inter-group structure at a coarser level than the full per-instance scatter. Use it when you want to see how groups relate to one another in RSHAP space rather than how individual instances are distributed.

**Note:** `coloring` and `categorical_colouring` are mutually exclusive. When `categorical_colouring` is set, `coloring` is ignored.

---

### Changing the aggregation function

The default aggregation is `np.sum`. Change this to shift the meaning of the axes:

```python
# Mean instead of sum — normalises for group size differences
rshap.cc_plot(cc_function=np.mean)

# Maximum — highlights the most extreme influence per instance
rshap.cc_plot(cc_function=np.max)
```

Any function that accepts a 2D array and an `axis=` keyword argument works.

---

### Axis reference lines

```python
# Default: orange horizontal line at y=0, red vertical line at x=0
rshap.cc_plot(axis_lines=True)

# Disable entirely
rshap.cc_plot(axis_lines=False)
```

The lines mark the boundary between positive and negative influence in each direction. Points in the upper-right quadrant are both hard to predict themselves (positive composition) and help predict others (positive contribution); points in the lower-left are easy to predict and make others easier too. See [INTERPRETATION.md](INTERPRETATION.md) for the full quadrant interpretation.

---

### Marker appearance — `drawparams`

Any keyword accepted by `plt.scatter` can be passed via `drawparams`:

```python
rshap.cc_plot(
    coloring=y,
    drawparams={
        'marker':     'o',      # marker shape
        's':          40,       # marker size (points²)
        'alpha':      0.7,      # transparency
        'edgecolors': 'none',   # no outline
        'linewidths': 0,
    }
)
```

Default is `{'marker': 'o'}` with matplotlib's default size and full opacity.

---

### Font size

```python
rshap.cc_plot(fontsizes=12)   # smaller labels
rshap.cc_plot(fontsizes=20)   # larger labels for presentations
```

Default is `16`. In group mode (`categorical_colouring`), axis labels are additionally scaled to `fontsizes × 1.2`.

---

### Saving the CC plot

```python
fig, ax = plt.subplots(figsize=(8, 6))
sc = rshap.cc_plot(coloring=y)
plt.colorbar(sc, label="Target")
plt.title("CC Plot — Airfoil Self-Noise")
fig.savefig("cc_plot.png", dpi=150, bbox_inches='tight')
plt.show()
```

Use `bbox_inches='tight'` whenever a legend or colorbar is present to prevent them being clipped at the figure edge.

---

## The contribution heatmap

### What it shows

`draw_heatmap` aggregates the contribution matrix into a group×group grid. Each cell shows the **mean signed contribution** from all instances in one group to all instances in another group.

- **Rows** (y-axis, labelled *Contribution*): the *giving* group — instances whose marginal addition to the training set is being measured
- **Columns** (x-axis, labelled *Composition*): the *receiving* group — instances whose prediction residuals are being changed

Cell `(i, j)` is positive when group i's instances consistently *improve* predictions for group j; negative when they consistently *worsen* them.

Values are normalised independently for positive and negative entries: the most positive cell maps to +1 and the most negative to −1. This means the full viridis range is always used regardless of the absolute magnitude of contributions.

---

### Basic usage

```python
from RSHAP import draw_heatmap
import matplotlib.pyplot as plt
import numpy as np

labels = np.array(["High quality"] * 855 + ["Low quality"] * 744)

fig, ax = draw_heatmap(rshap, labels)
plt.show()
```

`draw_heatmap` always returns `(fig, ax)`.

---

### Complete parameter reference

```python
draw_heatmap(
    rshap_object,
    labels,
    decimals  = 2,
    num_ticks = 9,
    fontsizes = 16,
    ax        = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rshap_object` | `ResidualDecompositionSymmetric` | required | A fitted RSHAP instance |
| `labels` | array-like | required | Group label for each of the N instances. Can be strings or integers. Must have length N. |
| `decimals` | int | `2` | Number of decimal places shown on colorbar tick labels |
| `num_ticks` | int | `9` | Number of evenly-spaced ticks on the colorbar |
| `fontsizes` | int | `16` | Font size for axis tick labels, axis labels, and colorbar label. Axis labels are scaled to `fontsizes × 1.2`. |
| `ax` | matplotlib Axes or `None` | `None` | Axes object to draw into. If `None`, a new `(6, 5)` figure and axes are created. |

**Returns:** `(fig, ax)`

---

### Tick label formatting

Axis tick labels alternate between two display formats to avoid overlapping text when there are many groups: even-indexed labels have ` -` appended, odd-indexed labels are shown plain. For example, with groups `["A", "B", "C", "D"]` the tick labels become `["A -", "B", "C -", "D"]`. This is handled automatically.

---

### Using an external axes object

Embed the heatmap inside a larger figure layout:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

draw_heatmap(rshap_ridge, quality_labels, ax=axes[0], fontsizes=13)
axes[0].set_title("Ridge regression")

draw_heatmap(rshap_xgb,   quality_labels, ax=axes[1], fontsizes=13)
axes[1].set_title("XGBoost")

fig.suptitle("Inter-class contributions — Wine Quality", fontsize=15)
fig.tight_layout()
fig.savefig("heatmap_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Saving the heatmap

```python
fig, ax = draw_heatmap(rshap, labels, fontsizes=14)
ax.set_title("Inter-class contributions", fontsize=14)
fig.savefig("heatmap.png", dpi=150, bbox_inches='tight')
```

---

## Common patterns and what they mean

### Tight central cluster with a few displaced points (CC plot)

The main cloud of instances shares similar composition and contribution scores — they all behave like average members of the dataset. The displaced points are anomalous: extreme positive composition indicates instances whose errors persist regardless of what training data surrounds them; extreme negative contribution indicates instances that degrade predictions for others when included.

### Two separated clusters by class color (CC plot)

Classes occupy distinct regions of RSHAP space. The model's residual structure is systematically different between classes — each class influences and is influenced differently. This is a sign the model has learned a genuine structural difference.

### Overlapping clusters by class color (CC plot)

The classes are indistinguishable in RSHAP space. The model is not reliably separating them, either because it is underpowered for the task, or because the classes genuinely share the same local influence structure and the separation is driven by features RSHAP is not capturing.

### Extreme-value instances displaced from origin (CC plot, continuous coloring)

When coloring by a continuous target or feature, instances at the extremes of that variable appear far from the origin. This shows that the variable of interest drives the model's residual structure — the instances where it is highest or lowest are the ones the model finds hardest to predict consistently.

### Strongly negative diagonal cells (heatmap)

Within-group influence is negative — instances help predict their own group-mates. This is the expected pattern in a well-fit model: adding more examples of a class or category makes the model better at predicting other members of the same group.

### Asymmetric off-diagonal cells (heatmap)

Cell `(A→B)` is much stronger in magnitude than `(B→A)`. Group A's instances have a disproportionate effect on group B's predictions relative to the reverse. This asymmetry often reflects imbalance in representation or difficulty: the majority class or the easier-to-predict group tends to exert more influence on the harder or minority group.

### Uniform heatmap (all cells similar magnitude)

All groups influence each other roughly equally. This can mean the groups are not meaningfully distinct in terms of model residuals, or that more RSHAP iterations are needed to resolve finer structure.
