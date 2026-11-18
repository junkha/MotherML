import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_regression
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import (
    ARDRegression,
    Lasso,
    LogisticRegression,
    MultiTaskLasso,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mother import ml
from mother.ml.models.m_lasso import LassoClassifierMother, LassoRegressorMother
from mother.optimization.core import MotherTuner


def _scaled(loader):
    X, y = loader(return_X_y=True)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("scale", [True, False])
def test_binary_matches_previous_liblinear_model(scale):
    X, y = load_breast_cancer(return_X_y=True)
    if scale:
        X = StandardScaler().fit_transform(X)
    previous = LogisticRegression(
        l1_ratio=1, solver="liblinear", C=1, class_weight="balanced", max_iter=3000, random_state=42
    ).fit(X, y)

    model = LassoClassifierMother().fit(X, y)

    assert model.solver_ == "liblinear"
    assert model.solver == "auto"
    np.testing.assert_array_equal(model.coef_, previous.coef_)
    np.testing.assert_array_equal(model.predict(X), previous.predict(X))


@pytest.mark.parametrize("loader", [load_iris, load_wine])
def test_multiclass_uses_saga_and_converges(loader):
    X, y = _scaled(loader)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LassoClassifierMother(model_type="classification_multiclass").fit(X, y)

    assert model.solver_ == "saga"
    assert model.solver == "auto"
    assert model.coef_.shape == (len(np.unique(y)), X.shape[1])
    assert set(model.predict(X)) <= set(np.unique(y))
    assert model.predict_proba(X).shape == (len(y), len(np.unique(y)))


@pytest.mark.parametrize("loader", [load_breast_cancer, load_iris, load_wine])
def test_coefficients_are_sparse(loader):
    X, y = _scaled(loader)
    model = LassoClassifierMother().fit(X, y)
    assert (model.coef_ == 0).any()


def test_user_solver_is_respected():
    X, y = _scaled(load_breast_cancer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LassoClassifierMother(solver="saga").fit(X, y)
    assert model.solver == "saga"
    assert model.solver_ == "saga"

    X, y = _scaled(load_iris)
    with pytest.raises(ValueError, match="liblinear"):
        LassoClassifierMother(solver="liblinear").fit(X, y)


@pytest.mark.parametrize("params", [{"solver": "unknown"}, {"model_type": "regression"}])
def test_invalid_parameters_raise(params):
    X, y = _scaled(load_iris)
    with pytest.raises(ValueError, match=next(iter(params))):
        LassoClassifierMother(**params).fit(X, y)


def test_target_type_is_single_target():
    model = LassoClassifierMother()
    assert model.target_type == "single_target"
    assert model.get_params()["target_type"] == "single_target"

    with pytest.raises(ValueError, match="must be 'single_target'"):
        LassoClassifierMother(target_type="multi_target")

    X, y = _scaled(load_breast_cancer)
    model = LassoClassifierMother().fit(X, y)
    with pytest.raises(ValueError, match="must be 'single_target'"):
        model.target_type = "multi_target"
    with pytest.raises(ValueError, match="must be 'single_target'"):
        model.set_params(target_type="multi_target")
    assert model.target_type == "single_target"

    y_multi = np.column_stack([y, y])
    with pytest.raises(ValueError, match="does not support multi-target"):
        LassoClassifierMother().fit(X, y_multi)


def test_params_round_trip():
    model = LassoClassifierMother(C=0.3, solver="saga", model_type="classification_multiclass")
    assert clone(model).get_params() == model.get_params()

    model.set_params(solver="auto", C=2.0)
    assert model.get_params()["solver"] == "auto"
    assert model.get_params()["C"] == 2.0

    X, y = _scaled(load_iris)
    fitted = LassoClassifierMother().fit(X, y)
    assert fitted.get_params()["solver"] == "auto"
    assert clone(fitted).get_params() == LassoClassifierMother().get_params()


def test_no_warning_on_construction(caplog):
    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        warnings.simplefilter("error")
        clone(LassoClassifierMother(model_type="classification_multiclass"))
    assert not caplog.records


@pytest.mark.parametrize("model_type", ["classification_binary", "classification_multiclass"])
def test_registry_returns_lasso_classifier_mother(model_type):
    assert ml.get_model_class_by_algorithm_and_type("lasso", model_type) is LassoClassifierMother


@pytest.mark.parametrize("loader", [load_breast_cancer, load_iris])
def test_tuner_tunes_c(loader):
    X, y = _scaled(loader)
    X, y = pd.DataFrame(X), pd.Series(y)
    model = LassoClassifierMother()
    tuner = MotherTuner(scorer="accuracy", n_trials_optuna=4, n_startup_trials=2)

    tuner.optimize(
        model,
        X=X,
        y=y,
        cross_validation=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        default_parameters=model.default_parameters(),
    )

    assert len(tuner.study.trials) == 4
    assert all("C" in trial.params for trial in tuner.study.trials)
    assert len({trial.params["C"] for trial in tuner.study.trials}) > 1


@pytest.fixture()
def multi_target_data():
    X, Y = make_regression(n_samples=200, n_features=30, n_informative=5, n_targets=3, noise=5, random_state=0)
    return pd.DataFrame(X), pd.DataFrame(Y, columns=["a", "b", "c"])


def test_regressor_single_target_matches_lasso(multi_target_data):
    X, Y = multi_target_data
    for y in (Y["a"], Y[["a"]]):
        model = LassoRegressorMother(alpha=1.0).fit(X, y)
        np.testing.assert_allclose(model.coef_, Lasso(alpha=1.0).fit(X, y).coef_)


def test_regressor_single_target_rejects_several_targets(multi_target_data):
    X, Y = multi_target_data
    with pytest.raises(ValueError, match="target_type='multi_target'"):
        LassoRegressorMother().fit(X, Y)


def test_regressor_single_target_supports_positive(multi_target_data):
    X, Y = multi_target_data
    model = LassoRegressorMother(alpha=1.0, positive=True).fit(X, Y["a"])
    assert (model.coef_ >= 0).all()


def test_regressor_multi_target_matches_multitask_lasso(multi_target_data):
    X, Y = multi_target_data
    weights = np.random.RandomState(0).rand(len(X))

    model = LassoRegressorMother(alpha=1.0, target_type="multi_target").fit(X, Y, sample_weight=weights)
    reference = MultiTaskLasso(alpha=1.0).fit(X, Y, sample_weight=weights)

    np.testing.assert_allclose(model.coef_, reference.coef_)
    np.testing.assert_allclose(model.predict(X), reference.predict(X))
    used = model.coef_ != 0
    assert (used == used[0]).all(), "MultiTaskLasso selects the same features for all targets"


@pytest.mark.parametrize(
    "params, target, match",
    [
        ({"target_type": "multi_target"}, "1d", "2-D target"),
        ({"target_type": "multi_target", "positive": True}, "2d", "positive"),
        ({"target_type": "unknown"}, "2d", "target_type"),
    ],
)
def test_regressor_invalid_multi_target_input_raises(multi_target_data, params, target, match):
    X, Y = multi_target_data
    with pytest.raises(ValueError, match=match):
        LassoRegressorMother(**params).fit(X, Y["a"] if target == "1d" else Y)


def test_regressor_params_round_trip():
    model = LassoRegressorMother(alpha=0.1, target_type="multi_target")
    assert clone(model).get_params() == model.get_params()
    assert model.set_params(target_type="single_target").get_params()["target_type"] == "single_target"


def test_get_and_set_params_follow_the_front_not_the_backend(multi_target_data, single_target_data):
    X, Y = multi_target_data
    front_keys = set(LassoRegressorMother().get_params())
    assert "ard_" not in front_keys

    lasso = LassoRegressorMother(alpha=0.2)
    before = lasso.get_params()
    lasso.fit(X, Y["a"])
    assert set(lasso.get_params()) == front_keys
    assert lasso.get_params() == before
    lasso.set_params(alpha=0.8, positive=True)
    assert lasso.get_params()["alpha"] == 0.8
    assert lasso.get_params()["positive"] is True
    lasso.fit(X, Y["a"])
    np.testing.assert_allclose(lasso.coef_, Lasso(alpha=0.8, positive=True).fit(X, Y["a"]).coef_)
    assert clone(lasso).get_params() == lasso.get_params()

    multi = LassoRegressorMother(alpha=0.2, target_type="multi_target")
    before = multi.get_params()
    multi.fit(X, Y)
    assert set(multi.get_params()) == front_keys
    assert multi.get_params() == before
    multi.set_params(alpha=0.4)
    multi.fit(X, Y)
    np.testing.assert_allclose(multi.coef_, MultiTaskLasso(alpha=0.4).fit(X, Y).coef_)
    multi.set_params(target_type="single_target", alpha=0.3)
    assert multi.get_params()["target_type"] == "single_target"
    multi.fit(X, Y["a"])
    assert np.ndim(multi.coef_) == 1
    np.testing.assert_allclose(multi.coef_, Lasso(alpha=0.3).fit(X, Y["a"]).coef_)

    Xs, ys = single_target_data
    ard = LassoRegressorMother(uncertainty=True, alpha_1=1e-6, threshold_lambda=1e4)
    before = ard.get_params()
    ard.fit(Xs, ys)
    assert set(ard.get_params()) == front_keys
    assert ard.get_params() == before
    ard.set_params(alpha_1=2e-4, alpha_2=3e-5, lambda_1=4e-5, lambda_2=5e-5, threshold_lambda=2e3)
    assert ard.get_params()["alpha_1"] == 2e-4
    assert ard.get_params()["threshold_lambda"] == 2e3
    ard.fit(Xs, ys)
    assert ard.ard_.alpha_1 == 2e-4
    assert ard.ard_.alpha_2 == 3e-5
    assert ard.ard_.lambda_1 == 4e-5
    assert ard.ard_.lambda_2 == 5e-5
    assert ard.ard_.threshold_lambda == 2e3
    ard.set_params(uncertainty=False, alpha=0.5)
    assert ard.get_params()["uncertainty"] is False
    assert ard.get_params()["alpha"] == 0.5
    ard.fit(Xs, ys)
    np.testing.assert_allclose(ard.coef_, Lasso(alpha=0.5).fit(Xs, ys).coef_)
    assert not hasattr(ard, "ard_")
    ard.set_params(uncertainty=True)
    with pytest.raises(NotFittedError):
        ard.predict_uncertainty(Xs)
    assert clone(ard).get_params() == ard.get_params()

    with pytest.raises(ValueError, match="Invalid parameter 'C'"):
        LassoRegressorMother().set_params(C=1.0)


def test_get_params_reports_ard_parameters_and_pickle_restores_the_fit(single_target_data):
    X, y = single_target_data
    ard_parameters = {
        "alpha_1": 2e-5,
        "alpha_2": 3e-5,
        "lambda_1": 4e-5,
        "lambda_2": 5e-5,
        "threshold_lambda": 2e3,
    }
    model = LassoRegressorMother(uncertainty=True, **ard_parameters).fit(X, y)

    params = model.get_params()
    assert params["uncertainty"] is True
    assert params["target_type"] == "single_target"
    for name, value in ard_parameters.items():
        assert params[name] == value
        assert getattr(model.ard_, name) == value

    restored = pickle.loads(pickle.dumps(model))
    assert isinstance(restored.ard_, ARDRegression)
    assert restored.get_params()["uncertainty"] is True
    for name, value in ard_parameters.items():
        assert restored.get_params()[name] == value
    pdt_columns = ["pred", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]
    original = model.predict_uncertainty(X).loc[:, pdt_columns]
    reloaded = restored.predict_uncertainty(X).loc[:, pdt_columns]
    pd.testing.assert_frame_equal(reloaded, original)


def test_clone_refit_repeats_lasso_multitask_and_ard(multi_target_data, single_target_data):
    X, Y = multi_target_data
    uncertainty_columns = ["pred", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]

    lasso = LassoRegressorMother(alpha=0.3, positive=True, max_iter=2000).fit(X, Y["a"])
    lasso_clone = clone(lasso)
    assert lasso_clone.get_params() == lasso.get_params()
    assert not hasattr(lasso_clone, "coef_")
    assert not hasattr(lasso_clone, "ard_")
    lasso_clone.fit(X, Y["a"])
    assert not hasattr(lasso_clone, "ard_")
    assert lasso_clone.positive is True
    np.testing.assert_allclose(lasso_clone.coef_, lasso.coef_)
    np.testing.assert_allclose(lasso_clone.predict(X), lasso.predict(X))

    multi = LassoRegressorMother(alpha=0.4, target_type="multi_target", tol=1e-5).fit(X, Y)
    multi_clone = clone(multi)
    assert multi_clone.get_params() == multi.get_params()
    assert multi_clone.get_params()["target_type"] == "multi_target"
    assert not hasattr(multi_clone, "coef_")
    multi_clone.fit(X, Y)
    assert not hasattr(multi_clone, "ard_")
    assert multi_clone.coef_.ndim == 2
    np.testing.assert_allclose(multi_clone.coef_, multi.coef_)
    np.testing.assert_allclose(multi_clone.predict(X), multi.predict(X))

    Xs, ys = single_target_data
    ard_parameters = {
        "alpha_1": 2e-5,
        "alpha_2": 3e-5,
        "lambda_1": 4e-5,
        "lambda_2": 5e-5,
        "threshold_lambda": 2e3,
    }
    ard = LassoRegressorMother(uncertainty=True, max_iter=400, **ard_parameters).fit(Xs, ys)
    ard_clone = clone(ard)
    assert ard_clone.get_params() == ard.get_params()
    assert ard_clone.get_params()["uncertainty"] is True
    assert not hasattr(ard_clone, "ard_")
    assert not hasattr(ard_clone, "coef_")
    ard_clone.fit(Xs, ys)
    assert isinstance(ard_clone.ard_, ARDRegression)
    assert ard_clone.ard_ is not ard.ard_
    for name, value in ard_parameters.items():
        assert ard_clone.get_params()[name] == value
        assert getattr(ard_clone.ard_, name) == value
    np.testing.assert_allclose(ard_clone.coef_, ard.coef_)
    pd.testing.assert_frame_equal(
        ard_clone.predict_uncertainty(Xs).loc[:, uncertainty_columns],
        ard.predict_uncertainty(Xs).loc[:, uncertainty_columns],
    )


def test_clone_refit_repeats_logistic_regression():
    X, y = _scaled(load_breast_cancer)
    binary = LassoClassifierMother(C=0.3, solver="auto", model_type="classification_binary").fit(X, y)
    binary_clone = clone(binary)
    assert binary_clone.get_params() == binary.get_params()
    assert binary_clone.get_params()["target_type"] == "single_target"
    assert not hasattr(binary_clone, "coef_")
    binary_clone.fit(X, y)
    assert binary_clone.solver_ == "liblinear"
    assert binary_clone.solver == "auto"
    np.testing.assert_allclose(binary_clone.coef_, binary.coef_)
    np.testing.assert_array_equal(binary_clone.predict(X), binary.predict(X))

    X, y = _scaled(load_iris)
    multiclass = LassoClassifierMother(C=0.7, solver="auto", model_type="classification_multiclass")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        multiclass.fit(X, y)
        multiclass_clone = clone(multiclass)
        assert multiclass_clone.get_params() == multiclass.get_params()
        assert not hasattr(multiclass_clone, "coef_")
        multiclass_clone.fit(X, y)
    assert multiclass_clone.solver_ == "saga"
    assert multiclass_clone.get_params()["target_type"] == "single_target"
    np.testing.assert_allclose(multiclass_clone.coef_, multiclass.coef_)
    np.testing.assert_array_equal(multiclass_clone.predict(X), multiclass.predict(X))


def test_classifier_get_and_set_params_follow_the_front():
    front_keys = set(LassoClassifierMother().get_params())
    assert "solver_" not in front_keys

    X, y = _scaled(load_breast_cancer)
    binary = LassoClassifierMother(C=0.4, solver="auto")
    before = binary.get_params()
    binary.fit(X, y)
    assert binary.solver_ == "liblinear"
    assert set(binary.get_params()) == front_keys
    assert binary.get_params() == before
    assert binary.get_params()["solver"] == "auto"
    binary.set_params(C=2.0, solver="saga")
    assert binary.get_params()["C"] == 2.0
    assert binary.get_params()["solver"] == "saga"
    binary.fit(X, y)
    assert binary.C == 2.0
    assert binary.solver_ == "saga"
    assert clone(binary).get_params() == binary.get_params()

    X, y = _scaled(load_iris)
    multiclass = LassoClassifierMother(model_type="classification_multiclass")
    before = multiclass.get_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        multiclass.fit(X, y)
    assert multiclass.solver_ == "saga"
    assert set(multiclass.get_params()) == front_keys
    assert multiclass.get_params() == before
    assert multiclass.get_params()["target_type"] == "single_target"

    with pytest.raises(ValueError, match="must be 'single_target'"):
        multiclass.set_params(C=0.2, target_type="multi_target")
    assert multiclass.get_params()["target_type"] == "single_target"
    assert multiclass.get_params()["C"] == before["C"]

    with pytest.raises(ValueError, match="Invalid parameter 'alpha'"):
        LassoClassifierMother().set_params(alpha=0.1)


@pytest.mark.parametrize("target_type", ["single_target", "multi_target"])
def test_tuner_tunes_alpha(multi_target_data, target_type):
    X, Y = multi_target_data
    model = LassoRegressorMother(target_type=target_type)
    tuner = MotherTuner(scorer="r2", n_trials_optuna=4, n_startup_trials=2)

    tuner.optimize(
        model,
        X=X,
        y=Y if target_type == "multi_target" else Y["a"],
        cross_validation=KFold(n_splits=3, shuffle=True, random_state=0),
        default_parameters=model.default_parameters(),
    )

    assert len(tuner.study.trials) == 4
    assert len({trial.params["alpha"] for trial in tuner.study.trials}) > 1


@pytest.fixture()
def single_target_data():
    X, y = make_regression(n_samples=300, n_features=40, n_informative=5, noise=5, random_state=0)
    index = range(100, 400)
    return pd.DataFrame(X, index=index), pd.Series(y, index=index)


def test_regressor_uncertainty_matches_ard(single_target_data):
    X, y = single_target_data
    model = LassoRegressorMother(uncertainty=True).fit(X, y)
    reference = ARDRegression(max_iter=1000, tol=1e-4).fit(X, y)
    mean, std = reference.predict(X, return_std=True)

    np.testing.assert_allclose(model.coef_, reference.coef_)
    np.testing.assert_allclose(model.predict(X), mean)

    result = model.predict_uncertainty(X)
    assert list(result.columns) == [
        "pred",
        "mean_predictions",
        "knowledge_uncertainty",
        "data_uncertainty",
        "total_uncertainty",
    ]
    assert result.index.equals(X.index)
    np.testing.assert_allclose(result["pred"], mean)
    np.testing.assert_allclose(result["total_uncertainty"], std)
    np.testing.assert_allclose(
        result["total_uncertainty"] ** 2, result["knowledge_uncertainty"] ** 2 + result["data_uncertainty"] ** 2
    )
    assert result["data_uncertainty"].nunique() == 1
    assert list(model.predict_uncertainty(X, uncertainty_for_opt=True).columns) == ["total_uncertainty"]


@pytest.mark.parametrize(
    "params, target, fit_kwargs, match",
    [
        ({"uncertainty": True, "target_type": "multi_target"}, "2d", {}, "ARDRegression only supports a single"),
        ({"uncertainty": True, "positive": True}, "1d", {}, "positive"),
        ({"uncertainty": True}, "1d", {"sample_weight": "ones"}, "sample_weight"),
        ({"uncertainty": True}, "2d", {}, "target_type='multi_target'"),
    ],
)
def test_regressor_uncertainty_invalid_combinations_raise(single_target_data, params, target, fit_kwargs, match):
    X, y = single_target_data
    target_values = pd.concat([y, y], axis=1) if target == "2d" else y
    if fit_kwargs.get("sample_weight") == "ones":
        fit_kwargs = {"sample_weight": np.ones(len(y))}
    with pytest.raises(ValueError, match=match):
        LassoRegressorMother(**params).fit(X, target_values, **fit_kwargs)


def test_regressor_uncertainty_defaults_and_space():
    lasso = LassoRegressorMother()
    ard = LassoRegressorMother(uncertainty=True)
    assert lasso.default_parameters("m__") == {"m__alpha": 1e-3}
    assert set(ard.default_parameters("m__")) == {
        "m__alpha_1",
        "m__alpha_2",
        "m__lambda_1",
        "m__lambda_2",
        "m__threshold_lambda",
    }
    assert clone(ard).get_params()["uncertainty"] is True


def test_tuner_tunes_ard_priors(single_target_data):
    X, y = single_target_data
    model = LassoRegressorMother(uncertainty=True)
    tuner = MotherTuner(scorer="r2", n_trials_optuna=4, n_startup_trials=2)

    tuner.optimize(
        model,
        X=X,
        y=y,
        cross_validation=KFold(n_splits=3, shuffle=True, random_state=0),
        default_parameters=model.default_parameters(),
    )

    assert all(
        set(trial.params) == {"alpha_1", "alpha_2", "lambda_1", "lambda_2", "threshold_lambda"}
        for trial in tuner.study.trials
    )
