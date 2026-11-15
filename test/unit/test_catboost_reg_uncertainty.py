import copy
import importlib.util
import inspect
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any, List
from unittest.mock import call, patch

import catboost
import numpy as np
import optuna
import pandas as pd
import pytest
import scipy.sparse
from catboost import CatBoostRegressor
from sklearn.base import clone, is_regressor
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mother.optimization as opt
from mother.ml.models.m_catboost import (
    CatboostGaussianProcessRegressorMother,
    CatboostRegressorMother,
)


def test_catboost_regressor_mother_preserves_positional_argument_order():
    """tune_loss_function must be appended after the pre-existing positional
    parameters (quantiles, data_uncertainty, model_type), not inserted before them,
    so existing positional construction keeps binding to the same options."""
    model = CatboostRegressorMother("single_target", True, False, None, True, "regression", False)

    assert model.target_type == "single_target"
    assert model.tune_tree_structure_type is True
    assert model.tune_boosting_type is False
    assert model.quantiles is None
    assert model.data_uncertainty is True
    assert model.model_type == "regression"
    assert model.tune_loss_function is False


@pytest.mark.slow
class TestCatboostModels(unittest.TestCase):
    def setUp(self) -> None:
        # Create synthetic data using make_blobs
        n_features = 5
        n_samples = 100

        # Generate synthetic data for regression
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        self.X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        self.y_regression = pd.Series(y, name="target")

        # Generate multi-target regression data
        self.y_multitarget_regression = pd.DataFrame({"0": y, "1": y})

        # Define quantiles for quantile regression tests
        self.quantiles: List[float] = [0.05, 0.5, 0.95]

    def _test_optim(self, model):
        cv = KFold(5)
        scorer = make_scorer(score_func=root_mean_squared_error, greater_is_better=False)

        # n_threads_optuna is the number of parallel threads for hyperparameter optimization
        # so it should not be higher then the number of cross-validation folds
        tuner = opt.MotherTuner(
            n_threads_optuna=5,
            scorer=scorer,
            n_trials_optuna=1,
            n_startup_trials=1,
            tuning_direction="maximize",
        )

        tuner.optimize(
            model,
            self.X,
            self.y_regression,
            cv,
            hyperparameter_space_function=model.get_hyperparameter_space,
            default_parameters=model.default_parameters(),
        )

    def test_CatboostRegressorMother_single_target_predict_with_uncertainty(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="single_target")
        model.fit(self.X, self.y_regression)
        result: pd.DataFrame = model.predict_uncertainty(self.X)
        self.assertIsInstance(result, pd.DataFrame)

        self.assertIn("pred", result.columns)
        self.assertIn("mean_predictions", result.columns)
        self.assertIn("knowledge_uncertainty", result.columns)
        self.assertIn("data_uncertainty", result.columns)
        self.assertIn("total_uncertainty", result.columns)

        self.assertTrue(np.issubdtype(result["pred"].dtype, np.number))
        self.assertTrue(np.issubdtype(result["mean_predictions"].dtype, np.number))
        self.assertTrue(np.issubdtype(result["knowledge_uncertainty"].dtype, np.number))
        self.assertTrue(result["data_uncertainty"].isna().all())
        self.assertTrue(result["total_uncertainty"].isna().all())

        correlation = result["mean_predictions"].corr(self.y_regression)
        self.assertGreater(correlation, 0.8, "Mean predictions are not sufficiently correlated with actual values.")

        self.assertTrue((result["knowledge_uncertainty"] >= 0).all(), "model uncertainty contains negative values.")

        self._test_optim(model)

    def test_CatboostRegressorMother_single_target_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="single_target")
        model.fit(self.X, self.y_regression, verbose=False)
        result: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)
        # Get results from predict_with_uncertainty
        result_with_uncertainty: pd.DataFrame = model.predict_uncertainty(self.X)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", result.columns)
        self.assertTrue(np.issubdtype(result["knowledge_uncertainty"].dtype, np.number))
        self.assertTrue((result["knowledge_uncertainty"] >= 0).all(), "model uncertainty contains negative values.")
        # Check if the values of "knowledge_uncertainty" are equal
        pd.testing.assert_series_equal(
            result["knowledge_uncertainty"],
            result_with_uncertainty["knowledge_uncertainty"],
            check_exact=True,
            obj="Model Uncertainty",
        )

    def test_CatboostRegressorMother_single_target_predict_uncertainty_passes_new_parameters(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="single_target")
        model.fit(self.X, self.y_regression, verbose=False)

        stub = pd.DataFrame(
            {
                "mean_predictions": np.zeros(len(self.X)),
                "knowledge_uncertainty": np.ones(len(self.X)),
                "data_uncertainty": np.nan,
                "total_uncertainty": np.nan,
            },
            index=self.X.index,
        )

        with patch("mother.ml.models.m_catboost.utils.get_virtual_prediction", return_value=stub) as mocked_predict:
            result = model.predict_uncertainty(self.X, n_ensembles=7, n_threads=3)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.X))
        self.assertIn("pred", result.columns)
        self.assertIn("knowledge_uncertainty", result.columns)
        mocked_predict.assert_called_once()
        self.assertEqual(mocked_predict.call_args.kwargs["virtual_ensembles_count"], 7)
        self.assertEqual(mocked_predict.call_args.kwargs["thread_count"], 3)

    def test_CatboostRegressorMother_predict_forwards_kwargs(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="single_target")
        model.fit(self.X, self.y_regression, verbose=False)

        mocked_output = np.ones(len(self.X))
        with patch.object(CatBoostRegressor, "predict", return_value=mocked_output) as mocked_predict:
            predictions = model.predict(self.X, thread_count=2, prediction_type="RawFormulaVal")

        self.assertIsInstance(predictions, np.ndarray)
        np.testing.assert_array_equal(predictions, mocked_output)
        mocked_predict.assert_called_once()
        self.assertIs(mocked_predict.call_args.kwargs["data"], self.X)
        self.assertEqual(mocked_predict.call_args.kwargs["thread_count"], 2)
        self.assertEqual(mocked_predict.call_args.kwargs["prediction_type"], "RawFormulaVal")

    def test_CatboostRegressorMother_multi_target_predict_with_uncertainty(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="multi_target")
        model.fit(self.X, self.y_multitarget_regression, verbose=False)
        result: pd.DataFrame = model.predict_uncertainty(self.X)
        self.assertIsInstance(result, pd.DataFrame)

        n_targets = self.y_multitarget_regression.shape[1]
        for idx in range(n_targets):
            self.assertIn(f"target_{idx}_mean_predictions", result.columns)
            self.assertIn(f"target_{idx}_knowledge_uncertainty", result.columns)
            self.assertIn(f"target_{idx}_data_uncertainty", result.columns)
            self.assertIn(f"target_{idx}_total_uncertainty", result.columns)
            self.assertTrue(np.issubdtype(result[f"target_{idx}_mean_predictions"].dtype, np.number))
            self.assertTrue(np.issubdtype(result[f"target_{idx}_knowledge_uncertainty"].dtype, np.number))
            self.assertIn(f"target_{idx}_pred", result.columns)
            self.assertTrue(result[f"target_{idx}_data_uncertainty"].isna().all())
            self.assertTrue(result[f"target_{idx}_total_uncertainty"].isna().all())

        # Check if the shape matches the input
        self.assertEqual(result.shape[0], self.X.shape[0])
        self._test_optim(model)

    def test_CatboostRegressorMother_multi_target_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="multi_target")
        model.fit(self.X, self.y_multitarget_regression, verbose=False)

        full_result: pd.DataFrame = model.predict_uncertainty(self.X)
        result_for_opt: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)

        ku_columns = [
            column
            for column in full_result.columns
            if column.startswith("target_") and column.endswith("_knowledge_uncertainty")
        ]
        expected_total = full_result[ku_columns].astype(float).max(axis=1)

        self.assertEqual(list(result_for_opt.columns), ["total_uncertainty"])
        self.assertEqual(len(result_for_opt), len(self.X))
        pd.testing.assert_series_equal(
            result_for_opt["total_uncertainty"],
            expected_total,
            check_names=False,
        )

    def test_CatboostRMSEWithUncertaintyRegression_predict(self):
        model: CatboostRegressorMother = CatboostRegressorMother(
            target_type="single_target", loss_function="RMSEWithUncertainty"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        mean_predictions: np.ndarray = model.predict(self.X)
        self.assertIsInstance(mean_predictions, np.ndarray)
        self.assertEqual(mean_predictions.shape, (self.X.shape[0],))
        self._test_optim(model)

    def test_CatboostRMSEWithUncertaintyRegression_predict_with_uncertainty(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            target_type="single_target", loss_function="RMSEWithUncertainty"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        predictions = model.predict_uncertainty(self.X)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertTrue(
            {"knowledge_uncertainty", "data_uncertainty", "total_uncertainty"}.issubset(predictions.columns)
        )
        self.assertEqual(predictions.shape[0], self.X.shape[0])

        self.assertTrue(
            (predictions["knowledge_uncertainty"] >= 0).all(), "model uncertainty contains negative values."
        )
        self.assertTrue((predictions["data_uncertainty"] >= 0).all(), "Data uncertainty contains negative values.")

        self.assertTrue(np.issubdtype(predictions["knowledge_uncertainty"].dtype, np.number))
        self.assertTrue(np.issubdtype(predictions["data_uncertainty"].dtype, np.number))
        self.assertTrue(np.issubdtype(predictions["total_uncertainty"].dtype, np.number))

    def test_CatboostRMSEWithUncertaintyRegression_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            target_type="single_target", loss_function="RMSEWithUncertainty"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        knowledge_uncertainty_df: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)
        self.assertIsInstance(knowledge_uncertainty_df, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", knowledge_uncertainty_df.columns)
        self.assertEqual(knowledge_uncertainty_df.shape[0], self.X.shape[0])

        self.assertTrue(np.issubdtype(knowledge_uncertainty_df["knowledge_uncertainty"].dtype, np.number))

        self.assertTrue(
            (knowledge_uncertainty_df["knowledge_uncertainty"] >= 0).all(),
            "model uncertainty contains negative values.",
        )

    def test_CatboostMultiQuantileRegression_predict(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(quantiles=self.quantiles.copy())
        model.fit(self.X, self.y_regression, verbose=False)
        preds: np.ndarray = model.predict(self.X)
        self.assertIsInstance(preds, np.ndarray, "Predict function did not return a NumPy array.")
        self.assertEqual(preds.shape, (self.X.shape[0],), "Median predictions do not have the correct shape.")

        # Standard output with new interface
        pred_uncertainty, quantile_array = model.predict_uncertainty(self.X, return_quantiles=True)
        self.assertIsInstance(quantile_array, np.ndarray, "Quantile array should be numpy array")
        # model._quantiles_processed includes DEFAULT_QUANTILES (0.25, 0.5, 0.75) merged with user-provided quantiles
        self.assertEqual(quantile_array.shape, (self.X.shape[0], len(model._quantiles_processed)))
        self.assertIsInstance(pred_uncertainty, pd.DataFrame)

        # Check shape: (n_samples, 5) for standard columns
        self.assertEqual(pred_uncertainty.shape[1], 5, "Should have 5 standard columns")

        # Check standard columns exist
        expected_columns = {
            "pred",
            "mean_predictions",
            "knowledge_uncertainty",
            "data_uncertainty",
            "total_uncertainty",
        }
        self.assertTrue(expected_columns.issubset(pred_uncertainty.columns), "Missing required columns")

        # Verify median prediction matches
        np.testing.assert_array_almost_equal(
            pred_uncertainty["pred"].values,
            preds,
            err_msg="Median predictions in 'pred' column do not match predict() output.",
        )

        self._test_optim(model)

    def test_CatboostMultiQuantileRegression_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            quantiles=self.quantiles.copy(),
        )
        model.fit(self.X, self.y_regression, verbose=False)
        result: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("total_uncertainty", result.columns)
        self.assertTrue(np.issubdtype(result["total_uncertainty"].dtype, np.number))
        self.assertTrue((result["total_uncertainty"] >= 0).all(), "total_uncertainty contains negative values.")

    def test_CatboostGaussianProcessRegressorMother_initialization(self) -> None:
        """Test model initialization and default parameters."""
        model: CatboostGaussianProcessRegressorMother = CatboostGaussianProcessRegressorMother()

        # Test scikit-learn estimator interface
        self.assertTrue(is_regressor(model))

        # Test default parameters
        params = model.get_params()
        self.assertEqual(params["samples"], 10)
        self.assertEqual(params["prior_iterations"], 100)
        self.assertEqual(params["model_type"], "regression")
        self.assertNotIn("tune_boosting_type", params)
        self.assertNotIn("tune_tree_structure_type", params)
        self.assertNotIn("tune_loss_function", params)

    def test_CatboostGaussianProcessRegressorMother_get_set_params(self) -> None:
        """Test parameter getting and setting."""
        model: CatboostGaussianProcessRegressorMother = CatboostGaussianProcessRegressorMother(
            samples=15,
            learning_rate=0.2,
            prior_iterations=500,
        )
        # Test get_params
        params = model.get_params()
        self.assertEqual(params["samples"], 15)
        self.assertAlmostEqual(params["learning_rate"], 0.2)
        self.assertEqual(params["prior_iterations"], 500)
        self.assertNotIn("tune_boosting_type", params)
        self.assertNotIn("tune_tree_structure_type", params)
        self.assertNotIn("tune_loss_function", params)
        self.assertFalse(model.tune_boosting_type)
        self.assertFalse(model.tune_tree_structure_type)
        self.assertFalse(model.tune_loss_function)

        # Test set_params
        model.set_params(samples=20, learning_rate=0.3, prior_iterations=600)
        params = model.get_params()
        self.assertEqual(params["samples"], 20)
        self.assertAlmostEqual(params["learning_rate"], 0.3)
        self.assertEqual(params["prior_iterations"], 600)

        # Test method chaining
        model2 = model.set_params(samples=25)
        self.assertIs(model2, model)
        self.assertEqual(model.get_params()["samples"], 25)

    def test_CatboostGaussianProcessRegressorMother_set_params_syncs_gp_params(self) -> None:
        """fit() reads learning_rate/max_depth/random_strength/random_score_type/verbose
        from self.gp_params, not from CatBoost's own _init_params -- set_params must keep
        gp_params in sync with these or Optuna trials silently have no effect."""
        model = CatboostGaussianProcessRegressorMother(learning_rate=0.1, max_depth=6, verbose=False)

        model.set_params(
            learning_rate=0.5, max_depth=3, random_strength=2.0, random_score_type="NormalWithModelSizeDecrease"
        )

        self.assertAlmostEqual(model.gp_params["learning_rate"], 0.5)
        self.assertEqual(model.gp_params["max_depth"], 3)
        self.assertAlmostEqual(model.gp_params["random_strength"], 2.0)
        self.assertEqual(model.gp_params["random_score_type"], "NormalWithModelSizeDecrease")

    def test_CatboostGaussianProcessRegressorMother_rejects_tune_kwargs_at_construction(self) -> None:
        """tune_boosting_type/tune_tree_structure_type/tune_loss_function are unsupported
        for GP posterior sampling; they must be rejected at construction, not silently
        forwarded to CatBoostRegressor via **kwargs."""
        with self.assertRaises(ValueError):
            CatboostGaussianProcessRegressorMother(tune_loss_function=True)

    def test_CatboostGaussianProcessRegressorMother_defaults_match_catboost_sampler(self) -> None:
        """Test that constructor and tuner defaults are the defaults of catboost's sample_gaussian_process."""
        sampler_defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(catboost.core.sample_gaussian_process).parameters.items()
        }
        expected = {
            "samples": sampler_defaults["samples"],
            "prior_iterations": sampler_defaults["prior_iterations"],
            "learning_rate": sampler_defaults["learning_rate"],
            "max_depth": sampler_defaults["depth"],
            "sigma": sampler_defaults["sigma"],
            "delta": sampler_defaults["delta"],
            "random_strength": sampler_defaults["random_strength"],
            "random_score_type": sampler_defaults["random_score_type"],
            "eps": sampler_defaults["eps"],
        }
        model = CatboostGaussianProcessRegressorMother()
        params = model.get_params()
        defaults = model.default_parameters()
        tuned = {"learning_rate", "prior_iterations", "sigma", "delta", "random_score_type"}
        for name, value in expected.items():
            self.assertEqual(params[name], value, name)
            if name in tuned:
                self.assertEqual(defaults[name], value, name)
            else:
                # Untuned parameters are left out so that the first tuning trial keeps the user's values
                self.assertNotIn(name, defaults)
        self.assertEqual(
            model._CATBOOST_DEFAULT_ITERATIONS,
            sampler_defaults["posterior_iterations"] + sampler_defaults["prior_iterations"],
        )
        # Settings that sample_gaussian_process fixes for every model it trains
        for name, value in {
            "loss_function": "RMSE",
            "bootstrap_type": "No",
            "boosting_type": "Plain",
            "grow_policy": "SymmetricTree",
        }.items():
            self.assertEqual(params[name], value, name)
            self.assertEqual(defaults[name], value, name)
        self.assertEqual(model.default_parameters(prefix="regressor__")["regressor__sigma"], sampler_defaults["sigma"])

    def test_CatboostGaussianProcessRegressorMother_forwards_seed_and_feature_types(self) -> None:
        """Test that random_seed and categorical features reach the sampler, as in CatBoostRegressor."""
        kwargs = {"samples": 2, "iterations": 150, "prior_iterations": 50}

        def fit_predict(**params: Any) -> np.ndarray:
            return (
                CatboostGaussianProcessRegressorMother(**kwargs, **params)
                .fit(self.X, self.y_regression)
                .predict(self.X)
            )

        np.testing.assert_allclose(fit_predict(random_seed=1), fit_predict(random_seed=1))
        np.testing.assert_allclose(fit_predict(), fit_predict(random_seed=0))
        self.assertFalse(np.allclose(fit_predict(random_seed=1), fit_predict(random_seed=2)))
        np.testing.assert_allclose(fit_predict(random_state=2), fit_predict(random_seed=2))

        X_categorical = self.X.copy()
        X_categorical["colour"] = np.where(self.X["feature_0"] > 0, "red", "blue")
        model = CatboostGaussianProcessRegressorMother(cat_features=["colour"], **kwargs)
        model.fit(X_categorical, self.y_regression)
        self.assertEqual(model.get_cat_feature_indices(), [X_categorical.columns.get_loc("colour")])
        self.assertTrue(np.isfinite(model.predict(X_categorical)).all())
        self.assertTrue(np.isfinite(model.predict_uncertainty(X_categorical)["knowledge_uncertainty"]).all())

    def test_CatboostGaussianProcessRegressorMother_search_space_keeps_posterior_trees(self) -> None:
        """Test that prior_iterations is only suggested below the configured number of trees."""
        study = optuna.create_study()
        for iterations, expected_high in ((None, 500), (300, 299), (40, 39)):
            params = {} if iterations is None else {"iterations": iterations}
            trial = study.ask()
            CatboostGaussianProcessRegressorMother(**params).get_hyperparameter_space(self.X, self.y_regression, trial)
            distribution = trial.distributions["prior_iterations"]
            self.assertEqual(distribution.high, expected_high, iterations)
            self.assertLessEqual(distribution.low, distribution.high)

    def test_CatboostGaussianProcessRegressorMother_search_space_only_tunes_used_parameters(self) -> None:
        """Test the GP search space: no bootstrap, tree structure, samples or depth, and valid ranges."""
        trial = optuna.create_study().ask()
        space = CatboostGaussianProcessRegressorMother().get_hyperparameter_space(self.X, self.y_regression, trial)

        tuned_parameters = {"learning_rate", "prior_iterations", "sigma", "delta", "random_score_type"}
        self.assertEqual(set(trial.params), tuned_parameters)
        self.assertEqual(set(space), tuned_parameters)
        self.assertEqual(
            (trial.distributions["learning_rate"].low, trial.distributions["learning_rate"].high), (0.01, 0.3)
        )
        self.assertEqual((trial.distributions["sigma"].low, trial.distributions["sigma"].high), (0.02, 0.5))
        self.assertEqual(
            set(trial.distributions["random_score_type"].choices), {"Gumbel", "NormalWithModelSizeDecrease"}
        )

        # Inside a pipeline every key carries the step prefix, so set_params reaches the GP estimator
        prefixed_trial = optuna.create_study().ask()
        prefixed = CatboostGaussianProcessRegressorMother().get_hyperparameter_space(
            self.X, self.y_regression, prefixed_trial, prefix="regressor__"
        )
        self.assertEqual(set(prefixed), {f"regressor__{name}" for name in tuned_parameters})

    def test_CatboostGaussianProcessRegressorMother_tuning_keeps_user_settings(self) -> None:
        """Test that a tuning trial changes only tuned parameters and every trial in the space can be fitted."""
        model = CatboostGaussianProcessRegressorMother(samples=3, max_depth=4, iterations=150, random_strength=0.05)
        study = optuna.create_study()
        study.enqueue_trial({"sigma": 0.02, "learning_rate": 0.3, "prior_iterations": 50})
        for _ in range(2):
            trial = study.ask()
            candidate = clone(model).set_params(**model.get_hyperparameter_space(self.X, self.y_regression, trial))
            self.assertEqual(
                {name: candidate.get_params()[name] for name in ("samples", "max_depth", "random_strength")},
                {"samples": 3, "max_depth": 4, "random_strength": 0.05},
            )
            candidate.fit(self.X, self.y_regression)
            self.assertEqual(len(candidate.posterior_tree_ranges_), 3)

    def test_CatboostGaussianProcessRegressorMother_invalid_model_type(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with self.assertRaises(ValueError):
            CatboostGaussianProcessRegressorMother(model_type="classification")

    def test_CatboostGaussianProcessRegressorMother_invalid_target_type(self) -> None:
        """Test that multi_target is rejected for GaussianProcess regressor wrapper."""
        with self.assertRaisesRegex(ValueError, "single_target"):
            CatboostGaussianProcessRegressorMother(target_type="multi_target")

    def test_CatboostGaussianProcessRegressorMother_fit_predict(self) -> None:
        """Test basic fit and predict functionality."""
        model = CatboostGaussianProcessRegressorMother(prior_iterations=50, samples=5, verbose=False)

        # Test fit returns self
        fit_result = model.fit(self.X, self.y_regression)
        self.assertIs(fit_result, model)

        # Test predict
        predictions = model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (len(self.X),))
        self.assertFalse(np.any(np.isnan(predictions)))

        # Test subset prediction
        subset_size = len(self.X) // 2
        subset_pred = model.predict(self.X.iloc[:subset_size])
        self.assertEqual(len(subset_pred), subset_size)

        # Test prediction on modified data
        X_new = self.X * 2
        new_predictions = model.predict(X_new)
        self.assertEqual(len(new_predictions), len(self.X))
        self.assertFalse(np.allclose(predictions, new_predictions))

    def test_CatboostGaussianProcessRegressorMother_predict_uncertainty(self) -> None:
        """Test uncertainty prediction functionality."""
        model = CatboostGaussianProcessRegressorMother(prior_iterations=50, samples=5, verbose=False)
        model.fit(self.X, self.y_regression)

        # Test predict_uncertainty
        results = model.predict_uncertainty(self.X)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("pred", results.columns)
        self.assertIn("mean_predictions", results.columns)
        self.assertIn("knowledge_uncertainty", results.columns)
        self.assertIn("data_uncertainty", results.columns)
        self.assertIn("total_uncertainty", results.columns)
        self.assertFalse(np.any(results["knowledge_uncertainty"] < 0))
        self.assertTrue(results["mean_predictions"].isna().all())
        self.assertTrue(results["data_uncertainty"].isna().all())
        self.assertTrue(results["total_uncertainty"].isna().all())
        self.assertEqual(len(results), len(self.X))

        # Test uncertainty_for_opt=True
        opt_uncertainty = model.predict_uncertainty(self.X, uncertainty_for_opt=True)
        self.assertIsInstance(opt_uncertainty, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", opt_uncertainty.columns)
        self.assertFalse(np.any(np.isnan(opt_uncertainty["knowledge_uncertainty"])))

        # Test that knowledge_uncertainty values are the same
        pd.testing.assert_series_equal(
            results["knowledge_uncertainty"],
            opt_uncertainty["knowledge_uncertainty"],
            check_exact=True,
            obj="Knowledge Uncertainty",
        )

    def test_CatboostGaussianProcessRegressorMother_predict_uncertainty_forwards_new_predict_parameters(self) -> None:
        """Test forwarding kwargs from predict_uncertainty into the per-sample tree-range predictions."""
        model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=10, verbose=False)
        model.fit(self.X, self.y_regression)
        X_small = self.X.iloc[:3]

        with patch.object(CatBoostRegressor, "predict", autospec=True, return_value=np.zeros(3)) as mock_predict:
            results = model.predict_uncertainty(X_small, thread_count=2, prediction_type="RawFormulaVal")

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", results.columns)
        self.assertEqual(len(results), len(X_small))
        self.assertEqual(
            mock_predict.call_args_list,
            [
                call(
                    model,
                    X_small,
                    ntree_start=ntree_start,
                    ntree_end=ntree_end,
                    thread_count=2,
                    prediction_type="RawFormulaVal",
                )
                for ntree_start, ntree_end in model.posterior_tree_ranges_
            ],
        )

    def test_CatboostGaussianProcessRegressorMother_predict_forwards_new_predict_parameters(self) -> None:
        """Test forwarding kwargs from predict into a single prediction of the merged model."""
        model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=10, verbose=False)
        model.fit(self.X, self.y_regression)
        X_small = self.X.iloc[:3]

        with patch.object(CatBoostRegressor, "predict", autospec=True, return_value=np.zeros(3)) as mock_predict:
            predictions = model.predict(X_small, thread_count=2, prediction_type="RawFormulaVal")

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(X_small))
        mock_predict.assert_called_once_with(model, X_small, thread_count=2, prediction_type="RawFormulaVal")

    def test_CatboostGaussianProcessRegressorMother_merged_model_reproduces_posterior_samples(self) -> None:
        """Test that the merged CatBoost model reproduces every posterior sample, their mean and their std."""
        sampled = {}
        sample_gaussian_process = catboost.core.sample_gaussian_process

        def capture_samples(*args, **kwargs):
            sampled["models"] = sample_gaussian_process(*args, **kwargs)
            return sampled["models"]

        model = CatboostGaussianProcessRegressorMother(samples=3, prior_iterations=20, verbose=False)
        with patch("catboost.core.sample_gaussian_process", side_effect=capture_samples):
            model.fit(self.X, self.y_regression)
        expected = np.array([sample.predict(self.X) for sample in sampled["models"]])

        self.assertEqual(model.tree_count_, sum(sample.tree_count_ for sample in sampled["models"]))
        self.assertEqual(model.posterior_tree_ranges_, [(0, 1000), (1000, 2000), (2000, 3000)])
        np.testing.assert_allclose(model.predict_posterior_samples(self.X), expected, atol=1e-9)
        np.testing.assert_allclose(model.predict(self.X), expected.mean(axis=0), atol=1e-9)
        np.testing.assert_allclose(
            model.predict_uncertainty(self.X)["knowledge_uncertainty"], expected.std(axis=0), atol=1e-9
        )

    def test_CatboostGaussianProcessRegressorMother_merge_handles_unequal_samples(self) -> None:
        """Test that samples with different tree counts, biases and scales are reconstructed from the merge."""
        samples = [
            CatBoostRegressor(iterations=iterations, random_seed=seed, verbose=False).fit(self.X, self.y_regression)
            for seed, iterations in enumerate([30, 70, 45])
        ]
        samples[1].set_scale_and_bias(0.5, 1.7)
        expected = np.array([sample.predict(self.X) for sample in samples])

        model = CatboostGaussianProcessRegressorMother()
        model._merge_posterior_samples(samples)

        self.assertEqual(model.posterior_tree_ranges_, [(0, 30), (30, 100), (100, 145)])
        np.testing.assert_allclose(model.predict_posterior_samples(self.X), expected, atol=1e-9)
        np.testing.assert_allclose(model.predict(self.X), expected.mean(axis=0), atol=1e-9)

    def test_CatboostGaussianProcessRegressorMother_save_and_load_model_keep_uncertainty(self) -> None:
        """Test that the posterior sample layout survives CatBoost's own save_model/load_model."""
        model = CatboostGaussianProcessRegressorMother(samples=3, prior_iterations=20, verbose=False)
        model.fit(self.X, self.y_regression)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "gp_catboost.cbm"
            model.save_model(str(path))
            loaded = CatboostGaussianProcessRegressorMother().load_model(str(path))

        self.assertEqual(loaded.posterior_tree_ranges_, model.posterior_tree_ranges_)
        pd.testing.assert_frame_equal(loaded.predict_uncertainty(self.X), model.predict_uncertainty(self.X))

    def test_CatboostGaussianProcessRegressorMother_predict_posterior_samples_rejects_tree_ranges(self) -> None:
        """Test that per-sample tree ranges cannot be overridden by the caller."""
        model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=10, verbose=False)
        model.fit(self.X, self.y_regression)

        with self.assertRaisesRegex(ValueError, "ntree_start and ntree_end"):
            model.predict_posterior_samples(self.X, ntree_end=10)

    def test_CatboostGaussianProcessRegressorMother_exponent_prediction_matches_posterior_samples(self) -> None:
        """Test that prediction_type='Exponent' transforms every sample, not the weighted tree sums."""
        sampled = {}
        sample_gaussian_process = catboost.core.sample_gaussian_process

        def capture_samples(*args, **kwargs):
            sampled["models"] = sample_gaussian_process(*args, **kwargs)
            return sampled["models"]

        model = CatboostGaussianProcessRegressorMother(samples=3, prior_iterations=20, verbose=False)
        with patch("catboost.core.sample_gaussian_process", side_effect=capture_samples):
            model.fit(self.X, self.y_regression)
        expected = np.array([sample.predict(self.X, prediction_type="Exponent") for sample in sampled["models"]])

        np.testing.assert_allclose(
            model.predict_posterior_samples(self.X, prediction_type="Exponent"), expected, rtol=1e-9
        )
        np.testing.assert_allclose(model.predict(self.X, prediction_type="Exponent"), expected.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(
            model.predict_uncertainty(self.X, prediction_type="Exponent")["knowledge_uncertainty"],
            expected.std(axis=0),
            rtol=1e-9,
        )

    def test_CatboostGaussianProcessRegressorMother_rejects_unsupported_prediction_type(self) -> None:
        """Test that prediction types without a per-sample reconstruction raise a clear error."""
        model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=10, verbose=False)
        model.fit(self.X, self.y_regression)

        for method in (model.predict, model.predict_uncertainty, model.predict_posterior_samples):
            with self.assertRaisesRegex(ValueError, "prediction_type 'Probability' is not supported"):
                method(self.X, prediction_type="Probability")

    def test_CatboostGaussianProcessRegressorMother_plain_catboost_model_has_no_uncertainty(self) -> None:
        """Test that a plain CatBoost model loaded into the class predicts but has no posterior samples."""
        plain = CatBoostRegressor(iterations=20, verbose=False).fit(self.X, self.y_regression)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "plain_catboost.cbm"
            plain.save_model(str(path))
            loaded = CatboostGaussianProcessRegressorMother().load_model(str(path))

        np.testing.assert_allclose(loaded.predict(self.X), plain.predict(self.X))
        with self.assertRaisesRegex(RuntimeError, "no posterior sample layout"):
            loaded.predict_uncertainty(self.X)

    def test_CatboostGaussianProcessRegressorMother_merge_requires_samples(self) -> None:
        """Test that merging an empty list of posterior samples raises."""
        with self.assertRaisesRegex(ValueError, "At least one posterior sample"):
            CatboostGaussianProcessRegressorMother()._merge_posterior_samples([])

    def test_CatboostGaussianProcessRegressorMother_merged_model_supports_catboost_api(self) -> None:
        """Test that the merged model can be used like a plain fitted CatBoostRegressor."""
        model = CatboostGaussianProcessRegressorMother(samples=3, iterations=200, prior_iterations=100)
        model.fit(self.X, self.y_regression)
        pool = catboost.Pool(self.X, self.y_regression)
        n_features = self.X.shape[1]

        self.assertIsInstance(model, CatBoostRegressor)
        self.assertEqual(model.feature_names_, list(self.X.columns))

        # Feature importances are available without an explicit call, as after CatBoost's own fit
        importances = model.feature_importances_
        self.assertEqual(importances.shape, (n_features,))
        np.testing.assert_allclose(importances, model.get_feature_importance())
        self.assertAlmostEqual(importances.sum(), 100.0)
        self.assertEqual(model.get_feature_importance(pool, type="LossFunctionChange").shape, (n_features,))
        prettified = model.get_feature_importance(prettified=True)
        self.assertEqual(set(prettified["Feature Id"]), set(self.X.columns))

        # SHAP values plus the expected value add up to the mean prediction
        shap_values = model.get_feature_importance(pool, type="ShapValues")
        self.assertEqual(shap_values.shape, (len(self.X), n_features + 1))
        np.testing.assert_allclose(shap_values.sum(axis=1), model.predict(self.X), atol=1e-8)

        # Every CatBoost feature importance type works with the same output shape as for a plain model
        expected_shapes: dict[str, tuple[int, ...]] = {
            "PredictionValuesChange": (n_features,),
            "LossFunctionChange": (n_features,),
            "FeatureImportance": (n_features,),
            "PredictionDiff": (n_features,),
            "SageValues": (n_features,),
            "Interaction": (n_features * (n_features - 1) // 2, 3),
            "ShapValues": (len(self.X), n_features + 1),
            "ShapInteractionValues": (len(self.X), n_features + 1, n_features + 1),
        }
        self.assertEqual(set(expected_shapes), {fstr_type.name for fstr_type in catboost.EFstrType})
        for fstr_type, shape in expected_shapes.items():
            data = self.X.iloc[:2] if fstr_type == "PredictionDiff" else pool
            values = np.asarray(model.get_feature_importance(data, type=fstr_type))
            self.assertEqual(values.shape, shape, fstr_type)
            self.assertTrue(np.isfinite(values).all(), fstr_type)
        shap_interactions = model.get_feature_importance(pool, type="ShapInteractionValues")
        np.testing.assert_allclose(shap_interactions[:, :-1, :-1].sum(axis=2), shap_values[:, :-1], atol=1e-8)

        self.assertIsInstance(model.score(self.X, self.y_regression), float)
        self.assertIn("RMSE", model.eval_metrics(pool, ["RMSE"]))
        np.testing.assert_allclose(
            list(model.staged_predict(self.X, eval_period=model.tree_count_))[-1], model.predict(self.X), atol=1e-8
        )
        self.assertEqual(model.calc_leaf_indexes(pool).shape, (len(self.X), model.tree_count_))
        self.assertEqual(len(model.get_tree_leaf_counts()), model.tree_count_)
        self.assertEqual(len(model.get_borders()), n_features)
        self.assertIn("mean_target", model.calc_feature_statistics(self.X, self.y_regression, feature=0, plot=False))

        # Columns are matched by name, as for a CatBoostRegressor fitted on a DataFrame
        np.testing.assert_allclose(model.predict(self.X[self.X.columns[::-1]]), model.predict(self.X))
        np.testing.assert_allclose(model.predict(self.X.to_numpy()), model.predict(self.X))
        np.testing.assert_allclose(model.predict(pool), model.predict(self.X))

        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.feature_names_, list(self.X.columns))
        np.testing.assert_allclose(restored.feature_importances_, importances)

        # shrink works on a copy and leaves the original model untouched
        shrunk = model.copy()
        shrunk.shrink(ntree_end=100)
        self.assertEqual(shrunk.tree_count_, 100)
        self.assertEqual(model.tree_count_, 600)

    def test_CatboostGaussianProcessRegressorMother_merged_model_exports_like_catboost(self) -> None:
        """Test that the merged model exports to CatBoost's formats and works with sklearn tooling."""
        model = CatboostGaussianProcessRegressorMother(samples=3, iterations=200, prior_iterations=100)
        model.fit(self.X, self.y_regression)
        expected = model.predict(self.X)

        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in ("cbm", "json"):
                path = Path(tmpdir) / f"gp.{fmt}"
                model.save_model(str(path), format=fmt)
                loaded = CatBoostRegressor().load_model(str(path), format=fmt)
                np.testing.assert_allclose(loaded.predict(self.X), expected, atol=1e-8, err_msg=fmt)
                self.assertEqual(loaded.feature_names_, list(self.X.columns), fmt)
                # The posterior sample layout travels with the file even when loaded as a plain CatBoost model
                self.assertIn(model._POSTERIOR_SAMPLES_METADATA_KEY, dict(loaded.get_metadata()), fmt)

            # The standalone Python export reproduces the predictions without CatBoost
            python_path = Path(tmpdir) / "gp_model.py"
            model.save_model(str(python_path), format="python")
            spec = importlib.util.spec_from_file_location("gp_model", python_path)
            exported = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exported)
            exported_predictions = np.array([exported.apply_catboost_model(list(row)) for row in self.X.to_numpy()])
            np.testing.assert_allclose(exported_predictions, expected, atol=1e-8)

            for fmt in ("cpp", "onnx", "pmml"):
                path = Path(tmpdir) / f"gp.{fmt}"
                model.save_model(str(path), format=fmt)
                self.assertGreater(path.stat().st_size, 0, fmt)

        # sklearn tooling clones and refits the estimator like any other regressor
        scores = cross_val_score(model, self.X, self.y_regression, cv=KFold(n_splits=2, shuffle=True, random_state=0))
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())

        pipeline = make_pipeline(
            StandardScaler(), CatboostGaussianProcessRegressorMother(samples=2, iterations=200, prior_iterations=100)
        )
        pipeline.fit(self.X, self.y_regression)
        self.assertEqual(pipeline.predict(self.X).shape, (len(self.X),))

    def test_CatboostGaussianProcessRegressorMother_handles_missing_values_like_catboost(self) -> None:
        """Test that missing feature values are supported as in CatBoostRegressor, missing targets are not."""
        X_missing = self.X.copy()
        X_missing.iloc[::7, 0] = np.nan
        X_missing.iloc[3::11, 2] = np.nan

        model = CatboostGaussianProcessRegressorMother(samples=3, prior_iterations=900, verbose=False)
        model.fit(X_missing, self.y_regression)

        self.assertTrue(np.isfinite(model.predict(X_missing)).all())
        self.assertTrue(np.isfinite(model.predict_posterior_samples(X_missing)).all())
        uncertainty = model.predict_uncertainty(X_missing)
        self.assertTrue(np.isfinite(uncertainty["knowledge_uncertainty"]).all())
        self.assertTrue(np.isfinite(model.get_feature_importance(catboost.Pool(X_missing, self.y_regression))).all())

        y_missing = self.y_regression.copy()
        y_missing.iloc[0] = np.nan
        with self.assertRaises(ValueError):
            CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=900).fit(self.X, y_missing)

    def test_CatboostGaussianProcessRegressorMother_accepts_sparse_input_like_catboost(self) -> None:
        """Test that scipy sparse matrices give the same model as dense input, and sparse arrays are rejected."""
        dense = self.X.to_numpy().copy()
        dense[np.abs(dense) < 0.7] = 0.0
        dense_model = CatboostGaussianProcessRegressorMother(samples=2, iterations=200, prior_iterations=100)
        dense_model.fit(dense, self.y_regression)

        for to_sparse in (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.coo_matrix):
            X_sparse = to_sparse(dense)
            sparse_model = CatboostGaussianProcessRegressorMother(samples=2, iterations=200, prior_iterations=100)
            sparse_model.fit(X_sparse, self.y_regression)

            self.assertEqual(sparse_model.posterior_tree_ranges_, dense_model.posterior_tree_ranges_)
            np.testing.assert_allclose(sparse_model.predict(X_sparse), dense_model.predict(dense))
            np.testing.assert_allclose(
                sparse_model.predict_posterior_samples(X_sparse), dense_model.predict_posterior_samples(dense)
            )
            np.testing.assert_allclose(
                sparse_model.predict_uncertainty(X_sparse)["knowledge_uncertainty"],
                dense_model.predict_uncertainty(dense)["knowledge_uncertainty"],
            )

        # CatBoost itself rejects scipy sparse arrays in fit and predict
        with self.assertRaises(TypeError):
            CatboostGaussianProcessRegressorMother(samples=2).fit(scipy.sparse.csr_array(dense), self.y_regression)

    def test_CatboostGaussianProcessRegressorMother_uses_configured_iterations(self) -> None:
        """Test that the number of trees per posterior sample follows CatBoost's iterations and its aliases."""
        for key in ("iterations", "n_estimators", "num_boost_round", "num_trees"):
            model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=50, **{key: 120})
            model.fit(self.X, self.y_regression)
            self.assertEqual(model.posterior_tree_ranges_, [(0, 120), (120, 240)], key)

        # Without an explicit value the GP uses CatBoost's own default number of trees
        plain = CatBoostRegressor(verbose=False).fit(self.X, self.y_regression)
        default_iterations = plain.get_all_params()["iterations"]
        self.assertEqual(CatboostGaussianProcessRegressorMother._CATBOOST_DEFAULT_ITERATIONS, default_iterations)
        model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=900)
        model.fit(self.X, self.y_regression)
        self.assertEqual(
            model.posterior_tree_ranges_, [(0, default_iterations), (default_iterations, 2 * default_iterations)]
        )

        model = CatboostGaussianProcessRegressorMother(samples=2).set_params(iterations=300, prior_iterations=100)
        model.fit(self.X, self.y_regression)
        self.assertEqual(model.posterior_tree_ranges_, [(0, 300), (300, 600)])

        with self.assertRaisesRegex(ValueError, "prior_iterations"):
            CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=200, iterations=200).fit(
                self.X, self.y_regression
            )

    def test_CatboostGaussianProcessRegressorMother_predict_before_fit_raises(self) -> None:
        """Test that predicting with an unfitted model raises a clear error."""
        model = CatboostGaussianProcessRegressorMother()

        for method in (model.predict, model.predict_uncertainty, model.predict_posterior_samples):
            with self.assertRaisesRegex(RuntimeError, "not been fitted"):
                method(self.X)

    def test_CatboostGaussianProcessRegressorMother_fit_rejects_multi_target_after_set_params(self) -> None:
        """Test fail-fast in fit when target_type is changed to unsupported value via set_params."""
        model = CatboostGaussianProcessRegressorMother(samples=2, prior_iterations=10, verbose=False)
        model.set_params(target_type="multi_target")

        with self.assertRaisesRegex(ValueError, "single_target"):
            model.fit(self.X, self.y_regression)

    def test_CatboostGaussianProcessRegressorMother_hyperparameter_optimization(self) -> None:
        """Test hyperparameter optimization using optuna."""
        model = CatboostGaussianProcessRegressorMother(
            prior_iterations=20,  # Reduced for faster testing
            samples=3,
            verbose=False,
        )

        # Test optimization
        self._test_optim(model)

    def test_CatboostGaussianProcessRegressorMother_pickling(self) -> None:
        """Test model serialization and deserialization."""
        model = CatboostGaussianProcessRegressorMother(samples=5, prior_iterations=50, learning_rate=0.1, verbose=False)
        model.fit(self.X, self.y_regression)

        # Get predictions before pickling
        results_before = model.predict_uncertainty(self.X)

        # Test pickling and unpickling
        pickled_model = pickle.dumps(model)
        unpickled_model = pickle.loads(pickled_model)

        # Test that parameters are preserved
        self.assertEqual(unpickled_model.get_params(), model.get_params())

        # Test that predictions are the same
        results_after = unpickled_model.predict_uncertainty(self.X)
        np.testing.assert_array_almost_equal(results_before["pred"], results_after["pred"])
        self.assertTrue(results_before["mean_predictions"].isna().all())
        self.assertTrue(results_after["mean_predictions"].isna().all())
        np.testing.assert_array_almost_equal(
            results_before["knowledge_uncertainty"], results_after["knowledge_uncertainty"]
        )
        self.assertEqual(unpickled_model.posterior_tree_ranges_, model.posterior_tree_ranges_)
        np.testing.assert_allclose(
            unpickled_model.predict_posterior_samples(self.X), model.predict_posterior_samples(self.X), atol=1e-12
        )

        # Retraining the unpickled model replaces its tree ranges and leaves the original untouched
        original_ranges = model.posterior_tree_ranges_
        unpickled_model.set_params(samples=3)
        unpickled_model.fit(self.X, self.y_regression)
        new_predictions = unpickled_model.predict(self.X)
        self.assertIsInstance(new_predictions, np.ndarray)
        self.assertEqual(len(unpickled_model.posterior_tree_ranges_), 3)
        self.assertEqual(model.posterior_tree_ranges_, original_ranges)

    def test_CatboostGaussianProcessRegressorMother_deepcopy_keeps_posterior_samples(self) -> None:
        """Test that copies keep the posterior sample layout and stay independent of the original."""
        model = CatboostGaussianProcessRegressorMother(samples=3, prior_iterations=900, verbose=False)
        model.fit(self.X, self.y_regression)
        original_ranges = model.posterior_tree_ranges_

        for copied in (copy.deepcopy(model), copy.copy(model), model.copy()):
            self.assertEqual(copied.posterior_tree_ranges_, original_ranges)
            np.testing.assert_allclose(
                copied.predict_posterior_samples(self.X), model.predict_posterior_samples(self.X), atol=1e-12
            )

        copied = copy.deepcopy(model)
        copied.set_params(samples=2)
        copied.fit(self.X, self.y_regression)
        self.assertEqual(len(copied.posterior_tree_ranges_), 2)
        self.assertEqual(model.posterior_tree_ranges_, original_ranges)

    def test_CatboostGaussianProcessRegressorMother_cloning(self) -> None:
        """Test model cloning functionality."""
        model = CatboostGaussianProcessRegressorMother(samples=15, prior_iterations=100, learning_rate=0.2)
        model.fit(self.X, self.y_regression)

        # Test cloning preserves parameters
        original_params = model.get_params()
        cloned_model = clone(model)
        self.assertEqual(cloned_model.get_params(), original_params)

        # A clone is unfitted and carries no posterior sample layout
        self.assertFalse(cloned_model.is_fitted())
        with self.assertRaises(RuntimeError):
            cloned_model.posterior_tree_ranges_

        # Test that cloned model can be modified independently
        cloned_model.set_params(samples=50)
        self.assertEqual(cloned_model.get_params()["samples"], 50)
        self.assertNotEqual(model.get_params()["samples"], 50)

        # Test that cloned model can be fitted independently
        cloned_model.fit(self.X, self.y_regression)
        predictions = cloned_model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(cloned_model.posterior_tree_ranges_), 50)
        self.assertEqual(len(model.posterior_tree_ranges_), 15)

    def test_CatboostGaussianProcessRegressorMother_state_persistence(self) -> None:
        """Test __getstate__ and __setstate__ methods."""
        model = CatboostGaussianProcessRegressorMother(samples=10, prior_iterations=50)
        model.fit(self.X, self.y_regression)

        # Test state saving and loading
        state = model.__getstate__()
        new_model = CatboostGaussianProcessRegressorMother()
        new_model.__setstate__(state)

        # Test that state is properly restored
        self.assertEqual(new_model.get_params(), model.get_params())

        # Test that restored model works
        predictions = new_model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
