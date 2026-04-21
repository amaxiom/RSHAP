import numpy as np
import pandas as pd
import math
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
import traceback
import copy
from joblib import Parallel, delayed

# Registry mapping string names to (regression_class, classification_class)
_MODEL_REGISTRY = {
    'ridge':    (Ridge,                   LogisticRegression),
    'logistic': (Ridge,                   LogisticRegression),
    'svm':      (SVR,                     SVC),
    'rf':       (RandomForestRegressor,   RandomForestClassifier),
    'mlp':      (MLPRegressor,            MLPClassifier),
    'xgb':      (XGBRegressor,            XGBClassifier),
}

def resolve_model(model_class, model_params, regression):
    """
    Resolve a string model name to the appropriate class, or pass through a class directly.
    Returns (model_class, model_params).
    """
    if isinstance(model_class, str):
        key = model_class.lower()
        if key not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_class}'. Choose from: {list(_MODEL_REGISTRY)}")
        reg_cls, clf_cls = _MODEL_REGISTRY[key]
        model_class = reg_cls if regression else clf_cls
    if model_params is None:
        model_params = {}
    return model_class, model_params

# Value function that creates a fresh model each time to avoid shared state issues
def __value_function(x, y, e_x, e_y, model):
    model.fit(x, y)
    return model.predict(e_x) - e_y

# Function that gets called in each process
# __val_func_classification(X[:0], Y[:0], X, Y, lr)
def __val_func_classification(x, y, e_x, e_y, model): 
    if len(np.unique(y)) == 1:
        temp = np.full_like(e_y, np.unique(y)[0] - e_y)
        
    elif len(np.unique(y)) != len(np.unique(e_y)):
        encoded_y = np.full(len(y), 0)
        label_count = 0
        label_map = {}

        for label in np.unique(y):
            encoded_y[y == label] = label_count
            label_map[label_count] = label
            label_count += 1

        model.fit(x, encoded_y)
        predicted = model.predict(e_x)
        decoded = np.array([label_map.get(p, np.nan) for p in predicted])
        temp = decoded - e_y
        
    else:
        model.fit(x, y)
        temp = model.predict(e_x) - e_y

    temp = np.nan_to_num(temp)  # replace NaN with 0
    temp[temp != 0] = 1
    return temp


# Function that gets called in each process
def _compute_marginal_contributions(X1, Y1, model_class, model_params, permutation, regression=True):
    '''
        Helper function for RSHAP, do not call
    '''
    try:
        N = X1.shape[0]
        phi = np.zeros((N, N))
        
        model = model_class(**model_params)
        subset = permutation[:1]
        Xs = X1[subset]
        Ys = Y1[subset]
        current_residuals = 0
        
        if regression:
            marginal_residuals = __value_function(Xs, Ys, X1, Y1, model)
        else:
            marginal_residuals = __val_func_classification(Xs, Ys, X1, Y1, model)
            
        phi[subset[0]] += marginal_residuals - current_residuals

        for i in range(1, N-1):
            subset = permutation[:i+1]
            Xs = X1[subset]
            Ys = Y1[subset]  

            current_residuals = marginal_residuals
            if regression:
                marginal_residuals = __value_function(Xs, Ys, X1, Y1, model)
            else:
                marginal_residuals = __val_func_classification(Xs, Ys, X1, Y1, model)
            phi[subset[i]] += marginal_residuals - current_residuals

        return phi.astype(np.float64)

    except Exception as e:
        import traceback
        with open("rshap_error.log", "a") as f:
            f.write("Subprocess crashed with error: " + str(e) + "\n")
            traceback.print_exc(file=f)
        print("Subprocess crashed with error:", e)
        print("Check rshap_error.log for details.")
        return None


def __get_heatmap(rshap_object, labels):
    contributions = rshap_object.get_contribution() 
    unique_labels = np.unique(labels)
    M = len(unique_labels)

    # Boolean mask: (M, N), where each row selects samples with a given instance group
    stacked_masks = np.stack([labels == morph for morph in unique_labels])  # (M, N)

    # This will generate the heatmap: (M, M)
    cc_heatmap = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            row_mask = stacked_masks[i]  # shape: (N,)
            col_mask = stacked_masks[j]  # shape: (N,)
            submatrix = contributions[np.ix_(row_mask, col_mask)]  # shape: (n_i, n_j)
            if submatrix.size > 0:
                cc_heatmap[i, j] = submatrix.mean()
                
    cc_heatmap[cc_heatmap > 0] /= np.max(cc_heatmap)
    cc_heatmap[cc_heatmap < 0] /= -np.min(cc_heatmap)

    return cc_heatmap, unique_labels


def draw_heatmap(rshap_object, labels, decimals=2, num_ticks=9, fontsizes=16, ax=None):
    """
    Visualises a heatmap of average contributions between different instance categories.
    Returns fig, ax for further manipulation (saving, etc).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cc_heatmap, unique_labels = __get_heatmap(rshap_object, labels)

    unique_elements_display = []
    counter = 0
    for i in np.unique(labels):
        if counter % 2 == 0:
            unique_elements_display.append(f'{i} -')
        else:
            unique_elements_display.append(str(i))
        counter += 1

    # Create axes if not supplied
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Use viridis colormap explicitly
    im = ax.imshow(cc_heatmap, cmap='viridis')

    ax.set_xticks(np.arange(0, len(unique_labels)))
    ax.set_yticks(np.arange(0, len(unique_labels)))
    ax.set_xticklabels(unique_elements_display, rotation=90, fontsize=fontsizes)
    ax.set_yticklabels(unique_elements_display, fontsize=fontsizes)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fontsizes)
    tick_locs = np.linspace(cbar.vmin, cbar.vmax, num_ticks)
    cbar.set_ticks(tick_locs)
    tick_labels = [f"{tick:.{decimals}f}" for tick in tick_locs]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Average Contribution', fontsize=fontsizes, labelpad=12)
    ax.set_xlabel("Composition", fontsize=fontsizes*1.2, labelpad=12)
    ax.set_ylabel("Contribution", fontsize=fontsizes*1.2, labelpad=12)

    fig.tight_layout()
    return fig, ax


class ResidualDecompositionSymmetric:
    def __init__(self, truncate=False):
        self.tracked_sum = []

    def fit(self, X1, Y1, model_class=None, model_params=None, iterations=100, regression=True, n_jobs=-1):
        """
        Fit the Residual Decomposition Symmetric explainer on the provided dataset.

        This method estimates individual contributions of each data point by averaging
        marginal residuals over multiple random orderings (permutations). It uses the
        specified model to compute residuals across each subset of the data.

        Parameters
        ----------
        X1 : array-like of shape (n_samples, n_features)
            The input features used to train the model.

        Y1 : array-like of shape (n_samples,)
            The target values corresponding to the input features.

        model_class : callable or str, default=None
            A class/constructor for the predictive model, or a string name from the model
            registry: 'ridge', 'logistic', 'svm', 'rf', 'mlp', 'xgb'.
            Must support fit(X, y) and predict(X) methods.

        model_params : dict, default=None
            Dictionary of parameters to initialize the model class. Passed as keyword arguments.

        iterations : int, default=100
            The number of permutations to use for computing Shapley-style residual contributions.
            More iterations yield smoother, more accurate estimates at the cost of computation time.
            Pass -1 to use the convergence method which automatically determines a stopping point based
            on the convergence rate.

        regression : bool, default=True
            Whether to use regression or classification objectives.

        n_jobs : int, default=-1
            Number of parallel jobs for joblib. -1 uses all available cores. Use 1 to disable
            parallelism (useful on memory-constrained systems or for debugging).
        """

        self.X1 = X1
        self.Y1 = Y1
        self.niter = iterations
        self.N = self.X1.shape[0]
        self.model_class, self.model_params = resolve_model(model_class, model_params, regression)
        self.r = regression
        self.n_jobs = n_jobs

        if iterations == -1:
            self.convergence_fit()
        else:
            self.iteration_fit()

    def iteration_fit(self):
        self.phi = np.zeros((self.N, self.N))
        indices = np.arange(0, self.N)

        permutations = []
        for _ in range(self.niter // 2):
            p = np.random.permutation(indices)
            permutations.append(p)
            permutations.append(p[::-1])

        args = [(self.X1, self.Y1, self.model_class, self.model_params, p, self.r) for p in permutations]

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_compute_marginal_contributions)(X, Y, m, mp, p, r) for (X, Y, m, mp, p, r) in tqdm(args)
        )
        
        for result in results:
            if result is not None:
                self.phi += result
            else:
                print("Warning: One parallel job returned None. Check previous error messages.")

        self.phi /= self.niter
        self.composition = self.phi
        
    def convergence_fit(self):
        self.phi = np.zeros((self.N, self.N))
        indices = np.arange(0, self.N)
        print("Using convergence fit")
        ncount = 0
        pcount = self.N 

        for i in range(0, 200):
            permutations = []
            
            for _ in range(pcount):
                p = np.random.permutation(indices)
                permutations.append(p)
                permutations.append(p[::-1])

            args = [(self.X1, self.Y1, self.model_class, self.model_params, p, self.r) for p in permutations]

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_compute_marginal_contributions)(X, Y, m, mp, p, r) for (X, Y, m, mp, p, r) in args
            )

            for result in results:
                self.phi += result
            
            ncount += pcount
            self.tracked_sum.append(copy.deepcopy(self.phi / ncount))
            
            if i >= 1:
                marginal_differences = np.mean(np.abs(1 - (self.tracked_sum[-1] / self.tracked_sum[-2])))
                if i % 10 == 0:
                    print(f"Marginal change at iteration {i+1} is {marginal_differences}")
                if marginal_differences < 0.025:
                    break

        self.phi /= ncount
        self.composition = self.phi

    def get_composition(self):
        return self.composition

    def get_contribution(self):
        summed = np.sum(self.composition, axis=0)
        return ((self.composition.T * -np.sign(summed))).T

    def cc_plot(self, coloring=None, fontsizes=16, axis_lines=True, cc_function=np.sum,
                categorical_colouring=None, drawparams={'marker': 'o'}, legend_label=None):
        """
        Plots a CC plot of composition sum vs. contribution sum, using viridis for all colorings.
    
        Parameters:
        -----------
        coloring : array-like or None
            Used for point coloring (continuous, discrete, or categorical).
        fontsizes : int
            Font size for axis labels.
        axis_lines : bool
            Whether to draw lines at x=0, y=0.
        cc_function : function
            Aggregation function for CC plot (e.g., np.sum, np.mean).
        categorical_colouring : array-like or None
            Used for grouping in plot.
        drawparams : dict
            Scatterplot customization.
        legend_label : str or None
            If provided, used as the legend title for discrete/categorical coloring.
    
        Returns:
        --------
        sc : scatter object(s)
        """
        viridis = plt.get_cmap('viridis') 
    
        if axis_lines:
            plt.axhline(0, color='orange')
            plt.axvline(0, color='red')
    
        x = cc_function(self.get_composition(), axis=0)
        y = cc_function(self.get_contribution(), axis=1)
        scatter_kwargs = dict(drawparams)
    
        legend_kwargs = dict(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=fontsizes, borderaxespad=0.)
    
        if categorical_colouring is None:
            plt.xlabel("Composition Sum", fontsize=fontsizes)
            plt.ylabel("Contribution Sum", fontsize=fontsizes)
            if coloring is None:
                coloring = self.Y1
    
            coloring = np.asarray(coloring)
    
            if coloring.dtype.kind in 'fc':  # continuous float/complex
                sc = plt.scatter(x, y, c=coloring, cmap=viridis, **scatter_kwargs)
                #cbar = plt.colorbar(sc, pad=0.02)
                # If you want to set the colorbar label outside this function:
                # cbar.set_label("...your label...", fontsize=fontsizes)
            elif coloring.dtype.kind in 'iu' or coloring.dtype.kind == 'O':
                categories = np.unique(coloring)
                colors = [viridis(i / max(1, len(categories) - 1)) for i in range(len(categories))]
                color_dict = {cat: colors[i] for i, cat in enumerate(categories)}
                sample_colors = [color_dict[cat] for cat in coloring]
                sc = plt.scatter(x, y, c=sample_colors, **scatter_kwargs)
                # Legend for discrete/categorical
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[cat],
                                      markersize=8, label=str(cat)) for cat in categories]
                if legend_label:
                    plt.legend(handles=handles, title=legend_label, **legend_kwargs).get_frame().set_linewidth(0)
                else:
                    plt.legend(handles=handles, **legend_kwargs).get_frame().set_linewidth(0)
            else:
                # fallback to categorical
                categories = np.unique(coloring)
                colors = [viridis(i / max(1, len(categories) - 1)) for i in range(len(categories))]
                color_dict = {cat: colors[i] for i, cat in enumerate(categories)}
                sample_colors = [color_dict[cat] for cat in coloring]
                sc = plt.scatter(x, y, c=sample_colors, **scatter_kwargs)
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[cat],
                                      markersize=10, label=str(cat)) for cat in categories]
                if legend_label:
                    plt.legend(handles=handles, title=legend_label, **legend_kwargs).get_frame().set_linewidth(0)
                else:
                    plt.legend(handles=handles, **legend_kwargs).get_frame().set_linewidth(0)
        else:
            sc = []
            categories = np.unique(categorical_colouring)
            colors = [viridis(i / max(1, len(categories) - 1)) for i in range(len(categories))]
            color_dict = {cat: colors[i] for i, cat in enumerate(categories)}
            for i, label in enumerate(categories):
                label_indices = np.where(categorical_colouring == label)[0]
                c_compositions = cc_function(self.get_composition()[label_indices], axis=0)
                c_contributions = cc_function(self.get_contribution()[:, label_indices], axis=1)
                s = plt.scatter(c_compositions, c_contributions, label=label, color=color_dict[label], **scatter_kwargs)
                sc.append(s)
            plt.xlabel("Composition Sum", fontsize=fontsizes*1.2, labelpad=12)
            plt.ylabel("Contribution Sum", fontsize=fontsizes*1.2, labelpad=12)
            # Legend for group
            if legend_label:
                plt.legend(title=legend_label, **legend_kwargs).get_frame().set_linewidth(0)
            else:
                plt.legend(**legend_kwargs).get_frame().set_linewidth(0)
    
        plt.tight_layout()
        return sc
