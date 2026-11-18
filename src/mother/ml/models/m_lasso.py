import logging
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
from optuna.trial import Trial
from sklearn.linear_model import (
    ARDRegression,
    Lasso,
    LogisticRegression,
    MultiTaskLasso,
)
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from mother.ml import properties as props
from mother.ml.core import AbstractMotherPipeline
from mother.ml.models import utils

module_logger: logging.Logger = logging.getLogger(__name__)


class LassoRegressorMother(Lasso, AbstractMotherPipeline):
    """
    MOTHER class for a LASSO regression including hyperparameter optimization.

    As in the CatBoost models, the target type is not detected from the data but set explicitly with
    ``target_type``; in a pipeline built from settings it is taken from ``settings.model.target_type``.
    A single-target model never returns several targets, and vice versa.

    - ``target_type="single_target"`` (default) fits a ``Lasso`` on one target. A target with more than
      one column raises a ``ValueError``.
    - ``target_type="multi_target"`` fits a ``MultiTaskLasso`` on a 2-D target: one joint L1/L2 penalty
      over all targets, so a feature is either used for all targets or for none.

    ``MultiTaskLasso`` does not support ``positive=True`` (non-negative coefficients) or ``precompute``.
    If several targets need non-negative coefficients or their own feature selection, fit one
    single-target model per target instead.

    Uncertainty: with ``uncertainty=True`` the model is fitted as an ``ARDRegression`` (Automatic
    Relevance Determination) instead of a ``Lasso``, analogous to ``data_uncertainty=True`` in
    ``CatboostRegressorMother``. ARD is a Bayesian sparse linear regression: every feature gets its own
    prior precision, and features whose precision exceeds ``threshold_lambda`` are pruned (coefficient
    0), so it selects features like Lasso, but without an ``alpha`` penalty.

    ``ARDRegression.predict(X, return_std=True)`` returns the standard deviation of the predictive
    distribution, which already contains both uncertainty types. Its variance for a sample ``x`` is

        total_variance(x) = x^T Sigma x + 1 / alpha_

    - ``x^T Sigma x``: ``Sigma`` (``sigma_``) is the posterior covariance of the coefficients, so this
      term is the coefficient uncertainty projected onto the sample. It differs per sample, grows for
      samples far from the training data and shrinks with more training data (epistemic).
    - ``1 / alpha_``: ``alpha_`` is the estimated precision of the noise, so this term is the noise
      variance (aleatoric). ARD assumes homoscedastic noise, so it is identical for all samples; unlike
      ``RMSEWithUncertainty`` in CatBoost, it does not vary per sample, and a ranking of samples by
      uncertainty therefore comes only from the knowledge part.

    ``predict_uncertainty`` returns these parts as standard deviations:

    - ``knowledge_uncertainty``: ``sqrt(x^T Sigma x)``, computed as
      ``sqrt(total_uncertainty**2 - data_uncertainty**2)`` so that scikit-learn's internal centring and
      feature pruning do not have to be repeated;
    - ``data_uncertainty``: ``sqrt(1 / alpha_)``;
    - ``total_uncertainty``: ``sqrt(knowledge_uncertainty**2 + data_uncertainty**2)``, the standard
      deviation returned by ``ARDRegression.predict(X, return_std=True)``.

    ``ARDRegression`` only supports a single target, sample weights are not supported, and there is no
    ``positive`` option, so ``uncertainty=True`` requires ``target_type="single_target"``,
    ``positive=False`` and no ``sample_weight``. With ``uncertainty=True`` the parameters ``alpha_1``,
    ``alpha_2``, ``lambda_1``, ``lambda_2`` and ``threshold_lambda`` are used and tuned instead of
    ``alpha``; ``max_iter``, ``tol``, ``fit_intercept`` and ``copy_X`` are passed on.

    Missing target values are not supported in any mode; rows with missing targets have to be removed
    or imputed beforehand.
    """

    _parameter_constraints: dict = {
        **Lasso._parameter_constraints,
        "target_type": [StrOptions({"single_target", "multi_target"})],
        "uncertainty": ["boolean"],
        **{
            name: ARDRegression._parameter_constraints[name]
            for name in ("alpha_1", "alpha_2", "lambda_1", "lambda_2", "threshold_lambda")
        },
    }

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        precompute: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        warm_start: bool = False,
        positive: bool = False,
        random_state: Optional[int] = None,
        selection: str = "cyclic",
        target_type: props.TargetType = "single_target",
        uncertainty: bool = False,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        threshold_lambda: float = 1e4,
    ) -> None:
        """
        Initialize LassoRegressorMother.

        Args:
            positive (bool): Force non-negative coefficients. Only supported with
                ``target_type="single_target"`` and ``uncertainty=False``.
            target_type (str): Set explicitly, it is not detected from the data.
                ``"single_target"`` (default) fits ``Lasso`` on one target column.
                ``"multi_target"`` fits ``MultiTaskLasso`` with a shared feature selection over all
                targets; it requires a 2-D target, ``positive=False`` and ``uncertainty=False``.
            uncertainty (bool): If ``True``, fit an ``ARDRegression`` instead of a ``Lasso`` so that
                ``predict_uncertainty`` returns predictive standard deviations. Only for
                ``target_type="single_target"``. Defaults to ``False``.
            alpha_1, alpha_2 (float): Shape and rate of the Gamma prior over the noise precision of
                ``ARDRegression``. Only used with ``uncertainty=True``.
            lambda_1, lambda_2 (float): Shape and rate of the Gamma prior over the weight precisions of
                ``ARDRegression``. Only used with ``uncertainty=True``.
            threshold_lambda (float): Features with a weight precision above this value are pruned by
                ``ARDRegression``. Only used with ``uncertainty=True``.
            The remaining arguments are passed to ``sklearn.linear_model.Lasso``.
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection,
        )
        self.target_type = target_type
        self.uncertainty = uncertainty
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.threshold_lambda = threshold_lambda

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the model with ``Lasso`` (``target_type="single_target"``), ``MultiTaskLasso``
        (``target_type="multi_target"``) or ``ARDRegression`` (``uncertainty=True``).

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                ``"single_target"``: a target vector or a 2-D target with one column.
                ``"multi_target"``: a 2-D target with one column per target.
            sample_weight: array-like, optional
                Sample weights. Not supported with ``uncertainty=True``.
            check_input: bool, optional
                Passed to ``Lasso.fit`` for single-target fitting.

        Returns:
            self
        """
        self._validate_params()
        if self.target_type == "multi_target":
            if self.uncertainty:
                raise ValueError(
                    "uncertainty=True is not supported with target_type='multi_target': "
                    "ARDRegression only supports a single target."
                )
            if self.positive:
                raise ValueError("positive=True is not supported with target_type='multi_target'.")
            if np.ndim(y) != 2:
                raise ValueError("target_type='multi_target' requires a 2-D target with one column per target.")
            # Drop a previous ARD fit so predict_uncertainty cannot reuse a stale ard_.
            if hasattr(self, "ard_"):
                del self.ard_
            return MultiTaskLasso.fit(self, X, y, sample_weight=sample_weight)

        if np.ndim(y) == 2 and np.shape(y)[1] > 1:
            raise ValueError(
                f"target_type='single_target' got a target with {np.shape(y)[1]} columns. "
                "Set target_type='multi_target' to fit several targets."
            )
        if not self.uncertainty:
            # Drop a previous ARD fit so predict_uncertainty cannot reuse a stale ard_.
            if hasattr(self, "ard_"):
                del self.ard_
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

        if self.positive:
            raise ValueError("positive=True is not supported with uncertainty=True.")
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported with uncertainty=True.")
        self.ard_ = ARDRegression(
            max_iter=self.max_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            threshold_lambda=self.threshold_lambda,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
        ).fit(X, np.ravel(y))
        # Lasso.predict only needs the linear coefficients, so the ARD mean prediction is reused as is.
        for attribute in ("coef_", "intercept_", "n_features_in_", "feature_names_in_", "n_iter_"):
            if hasattr(self.ard_, attribute):
                setattr(self, attribute, getattr(self.ard_, attribute))
        return self

    def predict_uncertainty(self, X, uncertainty_for_opt: bool = False, **kwargs):
        """
        Predict targets with uncertainty.

        With ``uncertainty=True`` the uncertainty comes from ``ARDRegression`` (standard deviations,
        see the class docstring). With ``uncertainty=False`` the generic fallback is used, which only
        returns predictions.

        Parameters:
            X: array-like
                Feature matrix.
            uncertainty_for_opt: bool, optional
                If ``True``, return only the ``total_uncertainty`` column.
            **kwargs:
                Passed to the generic fallback when ``uncertainty=False``.

        Returns:
            pd.DataFrame: Columns ``pred``, ``mean_predictions``, ``knowledge_uncertainty``,
            ``data_uncertainty`` and ``total_uncertainty``.
        """
        if not self.uncertainty:
            return super().predict_uncertainty(X, **kwargs)
        check_is_fitted(self, "ard_")
        pred, total = self.ard_.predict(X, return_std=True)
        data = np.full_like(total, np.sqrt(1.0 / self.ard_.alpha_))
        knowledge = np.sqrt(np.clip(total**2 - data**2, 0.0, None))
        index = X.index if isinstance(X, pd.DataFrame) else None
        result = pd.DataFrame(
            {
                "pred": pred,
                "mean_predictions": pred,
                "knowledge_uncertainty": knowledge,
                "data_uncertainty": data,
                "total_uncertainty": total,
            },
            index=index,
        )
        if uncertainty_for_opt:
            return result.loc[:, ["total_uncertainty"]]
        return result

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Define the hyperparameter search space: ``alpha`` for Lasso and MultiTaskLasso, the Gamma
        priors and ``threshold_lambda`` for ARD (``uncertainty=True``).

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                Target vector.
            trial: optuna.trial.Trial
                Optuna trial object for suggesting hyperparameters.
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing hyperparameter names and their suggested values.
        """
        if not self.uncertainty:
            space = {"alpha": trial.suggest_float(prefix + "alpha", 1e-6, 1e1, log=True)}
        else:
            space = {
                name: trial.suggest_float(prefix + name, 1e-9, 1e-3, log=True)
                for name in ("alpha_1", "alpha_2", "lambda_1", "lambda_2")
            }
            space["threshold_lambda"] = trial.suggest_float(prefix + "threshold_lambda", 1e2, 1e6, log=True)
        return utils.add_prefix_to_dict_keys(space, prefix=prefix)

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return the default hyperparameters: ``alpha=1e-3`` for Lasso and MultiTaskLasso, the
        scikit-learn defaults of ``ARDRegression`` for ``uncertainty=True``.

        Parameters:
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        if not self.uncertainty:
            return utils.add_prefix_to_dict_keys({"alpha": 1e-3}, prefix=prefix)
        return utils.add_prefix_to_dict_keys(
            {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6, "threshold_lambda": 1e4},
            prefix=prefix,
        )

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The parameters are those of ``LassoRegressorMother`` itself. Which class performs
        the fit — ``Lasso``, ``MultiTaskLasso`` or ``ARDRegression`` — does not change the
        parameter set, and the values given here are the ones the next ``fit`` passes to
        that class.

        Parameters:
            **params: Parameter names and values. Names are the constructor arguments,
                including ``target_type``, ``uncertainty`` and the ARD priors.
        """
        return super().set_params(**params)

    def get_params(self, deep=True) -> dict:
        """
        Return the parameters of this estimator.

        The result always contains the constructor parameters, so a fitted model still
        reports how it was configured. ``uncertainty`` and ``target_type`` say which
        class the next ``fit`` uses. When ``uncertainty`` is ``True`` the ARD priors are
        part of the same dictionary: ``alpha_1``, ``alpha_2``, ``lambda_1``, ``lambda_2``
        and ``threshold_lambda``. Those are the values passed to ``ARDRegression``.

        The fitted ``ard_`` object is not one of these parameters. ``clone`` copies only
        this dictionary, so the clone is unfitted; fitting it again uses ``uncertainty``
        and the ARD priors, or the Lasso parameters, exactly as the original did. Pickling
        the estimator keeps ``ard_`` itself, so loading the model restores the fitted ARD
        regression.
        """
        params = super().get_params(deep=deep)
        for name in (
            "target_type",
            "uncertainty",
            "alpha_1",
            "alpha_2",
            "lambda_1",
            "lambda_2",
            "threshold_lambda",
        ):
            params[name] = getattr(self, name)
        return params


class LassoClassifierMother(LogisticRegression, AbstractMotherPipeline):
    """
    MOTHER class for a LASSO (L1-regularised logistic regression) classification of binary and
    multiclass targets, including hyperparameter optimization.

    As in the other MOTHER models, the target type is not detected from the data but set explicitly
    with ``target_type``; in a pipeline built from settings it is taken from
    ``settings.model.target_type``. ``LogisticRegression`` fits one target column (binary or
    multiclass), so only ``target_type="single_target"`` is supported. A multi-target setting or a
    target with more than one column raises a ``ValueError``.

    With ``solver="auto"`` (default) the solver is chosen at fit time: ``liblinear`` for binary targets
    and ``saga`` for multiclass targets, the only scikit-learn solver that supports pure L1 for
    multinomial problems. The chosen solver is stored in ``solver_``; the ``solver`` parameter keeps
    the value passed by the user. An explicitly passed solver is used as given.

    ``saga`` converges much better on standardised features, so scale the input features (for example
    with a ``StandardScaler`` in the pipeline) for multiclass targets.
    """

    _parameter_constraints: dict = {
        **LogisticRegression._parameter_constraints,
        "solver": [StrOptions({"auto"} | LogisticRegression._parameter_constraints["solver"][0].options)],
        "model_type": [StrOptions({"classification_binary", "classification_multiclass"})],
        "target_type": [StrOptions({"single_target"})],
    }

    def __init__(
        self,
        *,
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Optional[Union[Mapping, str]] = "balanced",
        random_state: int = 42,
        solver: str = "auto",
        max_iter: int = 3000,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        model_type: props.ModelType = "classification_binary",
        target_type: props.TargetType = "single_target",
    ) -> None:
        """
        Initialize LassoClassifierMother.

        Args:
            solver (str): ``"auto"`` or a ``LogisticRegression`` solver that supports pure L1
                (``liblinear`` for binary targets, ``saga`` for binary and multiclass targets).
            model_type (str): ``"classification_binary"`` or ``"classification_multiclass"``. Kept for
                consistency with ``CatboostClassifierMother``; the solver is chosen from the target,
                not from ``model_type``.
            target_type (str): Set explicitly, it is not detected from the data. Only
                ``"single_target"`` is supported; multi-target classification is not.
            The remaining arguments are passed to ``sklearn.linear_model.LogisticRegression``.
        """
        # l1_ratio=1: pure L1 regularisation via the sklearn ≥1.8 API.
        # penalty='l1' was deprecated in sklearn 1.8 and will be removed in 1.10;
        # l1_ratio=1 is the replacement (analogous to LogisticRegressionCV).
        super().__init__(
            l1_ratio=1,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,  # type: ignore
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )
        self.model_type = model_type
        self.target_type = target_type

    @property
    def target_type(self) -> props.TargetType:
        """Only ``"single_target"`` is supported; assignment validates immediately."""
        return self._target_type

    @target_type.setter
    def target_type(self, value: props.TargetType) -> None:
        if value != "single_target":
            raise ValueError(
                "target_type for LassoClassifierMother must be 'single_target'. "
                "Multi-target classification is not supported."
            )
        self._target_type = value

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model. ``solver="auto"`` resolves to ``liblinear`` for binary and ``saga`` for
        multiclass targets.

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                A single target vector, or a 2-D target with one column.
            sample_weight: array-like, optional
                Sample weights.

        Returns:
            self
        """
        # Re-check via the property so a bypassed ``_target_type`` still fails at fit time.
        if self.target_type != "single_target":
            raise ValueError(
                "target_type for LassoClassifierMother must be 'single_target'. "
                "Multi-target classification is not supported."
            )
        self._validate_params()
        if np.ndim(y) == 2 and np.shape(y)[1] > 1:
            raise ValueError(
                f"target_type='single_target' got a target with {np.shape(y)[1]} columns. "
                "LassoClassifierMother does not support multi-target classification."
            )
        solver = self.solver
        if solver == "auto":
            solver = "saga" if type_of_target(y, input_name="y") == "multiclass" else "liblinear"
        self.solver_ = solver
        # LogisticRegression.fit reads self.solver and does not accept "auto".
        user_solver, self.solver = self.solver, solver
        try:
            super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.solver = user_solver
        return self

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Define the hyperparameter search space for Lasso classification.

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                Target vector.
            trial: optuna.trial.Trial
                Optuna trial object for suggesting hyperparameters.
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing hyperparameter names and their suggested values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "C": trial.suggest_float(prefix + "C", 1e-6, 1e1, log=True),
            },
            prefix=prefix,
        )

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return the default hyperparameters for the Lasso model.

        Parameters:
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        return utils.add_prefix_to_dict_keys({"C": 1e0}, prefix=prefix)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The parameters are those of ``LassoClassifierMother`` itself. The logistic
        regression fitted underneath does not change the parameter set. ``target_type``
        stays ``"single_target"``; a multi-target value raises ``ValueError`` and leaves
        the estimator unchanged.

        Parameters:
            **params: Parameter names and values. Names are the constructor arguments,
                including ``model_type`` and ``target_type``.
        """
        if "target_type" in params and params["target_type"] != "single_target":
            raise ValueError(
                "target_type for LassoClassifierMother must be 'single_target'. "
                "Multi-target classification is not supported."
            )
        return super().set_params(**params)

    def get_params(self, deep=True) -> dict:
        """
        Return the parameters of this estimator.

        The result is the constructor parameters of ``LassoClassifierMother``, including
        ``model_type`` and ``target_type``. It does not change after a binary or
        multiclass fit, and the solver stored for that fit (``solver_``) is not part of
        it: ``solver`` keeps the value passed by the user.
        """
        return super().get_params(deep=deep)
