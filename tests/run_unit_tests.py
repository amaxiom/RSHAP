import sys, warnings
sys.path.insert(0, r"C:\Users\Amanda\Favorites\Machine_Learning\SOFTWARE\RSHAP")
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import unittest, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from RSHAP import ResidualDecompositionSymmetric, resolve_model, _compute_marginal_contributions, draw_heatmap, _MODEL_REGISTRY

np.random.seed(0)
N, F = 20, 3
Xr = np.random.randn(N, F)
yr = (Xr @ [1.5, -1.0, 0.5] + np.random.randn(N) * 0.2).astype(np.float64)
Xc = np.random.randn(N, F)
yc = (Xc[:, 0] > 0).astype(int)

def _fit(X, y, model="ridge", regression=True, iters=4):
    r = ResidualDecompositionSymmetric()
    r.fit(X, y, model_class=model, iterations=iters, regression=regression, n_jobs=1)
    return r

class TestResolveModel(unittest.TestCase):
    def test_registry_has_expected_keys(self):
        for key in ("ridge", "logistic", "svm", "rf", "mlp", "xgb"):
            self.assertIn(key, _MODEL_REGISTRY)
    def test_ridge_regression(self):
        cls, params = resolve_model("ridge", None, True)
        self.assertIs(cls, Ridge); self.assertEqual(params, {})
    def test_ridge_returns_logistic_for_classification(self):
        cls, _ = resolve_model("ridge", None, False)
        self.assertIs(cls, LogisticRegression)
    def test_logistic_alias(self):
        cls, _ = resolve_model("logistic", None, False)
        self.assertIs(cls, LogisticRegression)
    def test_svm_regression(self):
        self.assertIs(resolve_model("svm", None, True)[0], SVR)
    def test_svm_classification(self):
        self.assertIs(resolve_model("svm", None, False)[0], SVC)
    def test_rf_regression(self):
        self.assertIs(resolve_model("rf", None, True)[0], RandomForestRegressor)
    def test_rf_classification(self):
        self.assertIs(resolve_model("rf", None, False)[0], RandomForestClassifier)
    def test_mlp_regression(self):
        self.assertIs(resolve_model("mlp", None, True)[0], MLPRegressor)
    def test_mlp_classification(self):
        self.assertIs(resolve_model("mlp", None, False)[0], MLPClassifier)
    def test_xgb_regression(self):
        self.assertIs(resolve_model("xgb", None, True)[0], XGBRegressor)
    def test_xgb_classification(self):
        self.assertIs(resolve_model("xgb", None, False)[0], XGBClassifier)
    def test_case_insensitive(self):
        self.assertIs(resolve_model("RF", None, True)[0], RandomForestRegressor)
        self.assertIs(resolve_model("SVM", None, False)[0], SVC)
        self.assertIs(resolve_model("XGB", None, True)[0], XGBRegressor)
    def test_passthrough_class_object(self):
        cls, params = resolve_model(Ridge, {"alpha": 2.0}, True)
        self.assertIs(cls, Ridge); self.assertEqual(params, {"alpha": 2.0})
    def test_none_params_defaults_to_empty_dict(self):
        _, params = resolve_model(Ridge, None, True)
        self.assertEqual(params, {})
    def test_explicit_params_preserved(self):
        _, params = resolve_model("svm", {"C": 5.0, "kernel": "rbf"}, True)
        self.assertEqual(params, {"C": 5.0, "kernel": "rbf"})
    def test_invalid_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_model("neural_net", None, True)
    def test_error_message_lists_valid_names(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_model("bad_model", None, True)
        self.assertIn("ridge", str(ctx.exception).lower())

class TestComputeMarginalContributions(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.perm = np.random.permutation(N)
    def test_returns_ndarray(self):
        result = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        self.assertIsInstance(result, np.ndarray)
    def test_shape_regression(self):
        result = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        self.assertEqual(result.shape, (N, N))
    def test_dtype_float64(self):
        result = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        self.assertEqual(result.dtype, np.float64)
    def test_no_nan_regression(self):
        result = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        self.assertFalse(np.any(np.isnan(result)))
    def test_shape_classification(self):
        result = _compute_marginal_contributions(Xc, yc, LogisticRegression, {}, self.perm, False)
        self.assertEqual(result.shape, (N, N))
    def test_no_nan_classification(self):
        result = _compute_marginal_contributions(Xc, yc, LogisticRegression, {}, self.perm, False)
        self.assertFalse(np.any(np.isnan(result)))
    def test_classification_values_are_integers(self):
        result = _compute_marginal_contributions(Xc, yc, LogisticRegression, {}, self.perm, False)
        self.assertTrue(np.all(result == result.astype(int)))
    def test_symmetric_pair_different(self):
        fwd = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        rev = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm[::-1], True)
        self.assertFalse(np.allclose(fwd, rev))
    def test_returns_none_on_model_crash(self):
        class CrashModel:
            def fit(self, X, y): raise RuntimeError("deliberate crash")
        result = _compute_marginal_contributions(Xr, yr, CrashModel, {}, self.perm, True)
        self.assertIsNone(result)
    def test_deterministic_with_fixed_permutation(self):
        r1 = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        r2 = _compute_marginal_contributions(Xr, yr, Ridge, {}, self.perm, True)
        np.testing.assert_array_equal(r1, r2)

class TestRSHAPAttributes(unittest.TestCase):
    def test_phi_exists_after_fit(self):
        r = _fit(Xr, yr); self.assertTrue(hasattr(r, "phi"))
    def test_composition_exists_after_fit(self):
        r = _fit(Xr, yr); self.assertTrue(hasattr(r, "composition"))
    def test_n_jobs_stored(self):
        r = ResidualDecompositionSymmetric()
        r.fit(Xr, yr, model_class="ridge", iterations=4, n_jobs=3)
        self.assertEqual(r.n_jobs, 3)
    def test_model_class_resolved(self):
        r = _fit(Xr, yr, model="ridge"); self.assertIs(r.model_class, Ridge)
    def test_model_params_default_empty(self):
        r = _fit(Xr, yr); self.assertEqual(r.model_params, {})
    def test_regression_flag_stored(self):
        r = _fit(Xc, yc, regression=False); self.assertFalse(r.r)

class TestRSHAPOutputs(unittest.TestCase):
    def setUp(self):
        self.r_reg = _fit(Xr, yr, regression=True)
        self.r_clf = _fit(Xc, yc, regression=False)
    def test_composition_shape_regression(self):
        self.assertEqual(self.r_reg.get_composition().shape, (N, N))
    def test_contribution_shape_regression(self):
        self.assertEqual(self.r_reg.get_contribution().shape, (N, N))
    def test_composition_shape_classification(self):
        self.assertEqual(self.r_clf.get_composition().shape, (N, N))
    def test_phi_not_all_zeros_regression(self):
        self.assertFalse(np.all(self.r_reg.phi == 0))
    def test_phi_not_all_zeros_classification(self):
        self.assertFalse(np.all(self.r_clf.phi == 0))
    def test_no_nan_composition(self):
        self.assertFalse(np.any(np.isnan(self.r_reg.get_composition())))
    def test_no_nan_contribution(self):
        self.assertFalse(np.any(np.isnan(self.r_reg.get_contribution())))
    def test_composition_equals_phi(self):
        np.testing.assert_array_equal(self.r_reg.get_composition(), self.r_reg.phi)
    def test_contribution_rows_scaled_by_neg_sign_of_col_sums(self):
        # get_contribution() transposes, multiplies rows by -sign(col_sum), then transposes back.
        # Net effect: contrib[i, j] = comp[i, j] * -sign(sum_k comp[k, i])
        # i.e. every element in row i is scaled by -sign(col_sum of column i).
        comp    = self.r_reg.get_composition()
        contrib = self.r_reg.get_contribution()
        col_sums = np.sum(comp, axis=0)                        # shape (N,)
        expected = comp * (-np.sign(col_sums))[:, np.newaxis]  # broadcast over columns
        np.testing.assert_allclose(contrib, expected, atol=1e-10)
    def test_class_object_accepted(self):
        r = ResidualDecompositionSymmetric()
        r.fit(Xr, yr, model_class=Ridge, model_params={"alpha": 0.5}, iterations=4, n_jobs=1)
        self.assertEqual(r.get_composition().shape, (N, N))
    def test_more_iterations_changes_phi(self):
        r4 = _fit(Xr, yr, iters=4)
        r20 = _fit(Xr, yr, iters=20)
        self.assertFalse(np.allclose(r4.phi, r20.phi))
    def test_convergence_fit_runs(self):
        r = ResidualDecompositionSymmetric()
        r.fit(Xr, yr, model_class="ridge", iterations=-1, n_jobs=1)
        self.assertEqual(r.get_composition().shape, (N, N))

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.rshap = _fit(Xr, yr)
        self.labels = np.array(["A"] * (N // 2) + ["B"] * (N // 2))
    def tearDown(self):
        plt.close("all")
    def test_draw_heatmap_returns_fig_and_ax(self):
        fig, ax = draw_heatmap(self.rshap, self.labels)
        self.assertIsNotNone(fig); self.assertIsNotNone(ax)
    def test_draw_heatmap_accepts_external_ax(self):
        fig0, ax0 = plt.subplots()
        fig, ax = draw_heatmap(self.rshap, self.labels, ax=ax0)
        self.assertIs(ax, ax0)
    def test_cc_plot_no_coloring(self):
        plt.figure(); sc = self.rshap.cc_plot(); self.assertIsNotNone(sc)
    def test_cc_plot_continuous_coloring(self):
        plt.figure(); sc = self.rshap.cc_plot(coloring=yr); self.assertIsNotNone(sc)
    def test_cc_plot_categorical_coloring(self):
        plt.figure(); sc = self.rshap.cc_plot(coloring=self.labels); self.assertIsNotNone(sc)
    def test_cc_plot_group_coloring(self):
        plt.figure(); sc = self.rshap.cc_plot(categorical_colouring=self.labels); self.assertIsNotNone(sc)

suite = unittest.TestSuite()
loader = unittest.TestLoader()
for tc in (TestResolveModel, TestComputeMarginalContributions, TestRSHAPAttributes, TestRSHAPOutputs, TestVisualization):
    suite.addTests(loader.loadTestsFromTestCase(tc))
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
