"""Uncertainty-interface tests for NODE estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# Skip the entire module when the optional NODE/heads dependencies are absent.
pytest.importorskip("skorch")
torch = pytest.importorskip("torch")

from mother.ml.models.m_node import NODEClassifier, NODERegressor  # noqa: E402

# - serial: avoid PyTorch multiprocessing issues under pytest-xdist
# - slow: NODE training is computationally expensive
pytestmark = [pytest.mark.serial, pytest.mark.slow]

REQUIRED_UNCERTAINTY_COLS = {
    "pred",
    "mean_predictions",
    "knowledge_uncertainty",
    "data_uncertainty",
    "total_uncertainty",
}


def _classification_data():
    """Small breast-cancer split as float32 numpy arrays (skorch-friendly)."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _regression_data():
    """Small diabetes split as float32 numpy arrays (skorch-friendly)."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_predict_uncertainty_classification_node():
    """NODE classifiers return the standard predict_uncertainty() DataFrame format."""
    X_train, X_test, y_train, _ = _classification_data()
    model = NODEClassifier(num_trees=16, max_epochs=3, device="cpu", verbose=0)
    model.fit(X_train, y_train)
    pred = model.predict_uncertainty(X_test)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing classification uncertainty columns: {sorted(missing_cols)}"
    assert pred["total_uncertainty"].notna().all(), "total_uncertainty should be populated for classifiers"


def test_predict_uncertainty_regression_node():
    """NODE regressors return the standard predict_uncertainty() DataFrame format."""
    X_train, X_test, y_train, _ = _regression_data()
    model = NODERegressor(num_trees=16, max_epochs=3, device="cpu", verbose=0)
    model.fit(X_train, y_train)
    pred = model.predict_uncertainty(X_test)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"


def test_node_flow_uncertainty_columns_present():
    """NODE flow head returns the standard uncertainty columns."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )

    reg.fit(X_train, y_train)
    pred = reg.predict_uncertainty(X_test, num_samples=200, num_mc_samples=8)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"
    assert pred["data_uncertainty"].notna().all()
    assert pred["total_uncertainty"].notna().all()


def test_node_flow_uncertainty_decomposition_identity():
    """NODE flow head keeps total = data + knowledge uncertainty."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    pred = reg.predict_uncertainty(X_test, num_samples=200, num_mc_samples=8)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"

    knowledge = pred["knowledge_uncertainty"].to_numpy(dtype=float)
    data = pred["data_uncertainty"].to_numpy(dtype=float)
    total = pred["total_uncertainty"].to_numpy(dtype=float)

    # Epistemic (mutual information) is populated and non-negative; identity holds exactly.
    assert pred["knowledge_uncertainty"].notna().all()
    assert (knowledge >= -1e-6).all()
    np.testing.assert_allclose(total, data + knowledge, atol=1e-5)


def test_node_flow_quantiles_available():
    """NODE flow head returns predictive quantiles."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    q = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9], num_samples=200)
    assert q.shape == (len(X_test), 3)
    # Quantiles are monotonically non-decreasing per row.
    assert (np.diff(q, axis=1) >= -1e-4).all()


def test_flow_quantile_levels_and_cost(monkeypatch):
    """predict_quantiles returns exactly the requested levels without the uncertainty decomposition."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()
    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.1,
        num_trees=8,
        depth=2,
        max_epochs=1,
        device="cpu",
        verbose=0,
    ).fit(X_train, y_train)

    _, merged = reg.predict_uncertainty(X_test[:8], return_quantiles=True, quantiles=[0.9, 0.1], num_samples=20)
    assert merged.shape == (8, 5)  # sorted({0.1, 0.9} | {0.25, 0.5, 0.75})

    def fail(*args, **kwargs):
        raise AssertionError("predict_quantiles must not run the combined-uncertainty decomposition")

    monkeypatch.setattr(reg, "predict_with_combined_uncertainty", fail)
    assert reg.predict_quantiles(X_test[:8], quantiles=[0.9, 0.1], num_samples=20).shape == (8, 2)


def test_predict_uncertainty_warns_when_all_dropouts_zero():
    """NODE emits a warning when MC-dropout uncertainty is requested with dropout=0."""
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="subset",
        input_dropout=0.0,
        tree_dropout=0.0,
        num_trees=16,
        num_layers=1,
        depth=3,
        max_epochs=4,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    with pytest.warns(UserWarning, match="MC-dropout repeats are deterministic"):
        _ = reg.predict_uncertainty(X_test, num_samples=16)


def test_classifier_predict_uncertainty_warns_when_all_dropouts_zero():
    X_train, X_test, y_train, _ = _classification_data()
    clf = NODEClassifier(
        input_dropout=0.0, tree_dropout=0.0, num_trees=8, depth=2, max_epochs=1, device="cpu", verbose=0
    ).fit(X_train, y_train)

    with pytest.warns(UserWarning, match="MC-dropout repeats are deterministic"):
        clf.predict_uncertainty(X_test[:8], num_samples=4)


def test_flow_dropout_pred_column_is_predict_mode(monkeypatch):
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()
    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.1,
        num_trees=8,
        depth=2,
        max_epochs=1,
        device="cpu",
        verbose=0,
    ).fit(X_train, y_train)
    mode = np.arange(8, dtype=np.float32)
    monkeypatch.setattr(reg, "predict", lambda X, num_samples=1000: mode)

    frame = reg.predict_uncertainty(X_test[:8], num_samples=4, num_flow_samples=8)

    np.testing.assert_array_equal(frame["pred"], mode)
    assert not np.array_equal(frame["mean_predictions"], mode)


def test_predict_uncertainty_dropout_overrides_are_temporary():
    """Inference-only input/tree dropout overrides enable MC uncertainty safely."""
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="subset",
        input_dropout=0.0,
        tree_dropout=0.0,
        num_trees=16,
        num_layers=1,
        depth=3,
        max_epochs=4,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    result = reg.predict_uncertainty(X_test, num_samples=16, input_dropout=0.1, tree_dropout=0.1)

    assert result["knowledge_uncertainty"].mean() > 0
    assert reg.input_dropout == 0.0
    assert reg.tree_dropout == 0.0
    assert reg.module_.input_dropout == 0.0
    assert reg.module_.tree_dropout == 0.0

    with pytest.raises(ValueError, match="input_dropout must be in"):
        reg.predict_uncertainty(X_test, input_dropout=1.0)


def test_node_flow_balsa_emd_opt_signal_and_total_nan():
    """BALSA-EMD mode exposes epistemic score for optimisation and marks total as NaN."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    pred = reg.predict_uncertainty(
        X_test,
        num_samples=120,
        knowledge_method="balsa_emd",
    )

    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"
    assert pred["knowledge_uncertainty"].notna().all()
    assert pred["total_uncertainty"].isna().all()

    opt_signal = reg.predict_uncertainty(
        X_test,
        num_samples=120,
        knowledge_method="balsa_emd",
        uncertainty_for_opt=True,
    )
    assert list(opt_signal.columns) == ["knowledge_uncertainty"]
    opt_vals = opt_signal["knowledge_uncertainty"].to_numpy(dtype=float)
    pred_vals = pred["knowledge_uncertainty"].to_numpy(dtype=float)
    assert opt_vals.shape == pred_vals.shape
    assert np.isfinite(opt_vals).all()
    assert (opt_vals >= 0).all()


@pytest.fixture
def tiny_regression_data():
    features = np.random.default_rng(42).normal(size=(24, 3)).astype(np.float32)
    targets = features[:, 0] + 0.25 * features[:, 1]
    return features, targets


@pytest.mark.parametrize(
    ("input_rate", "tree_rate", "head_only"),
    [(0.5, 0.0, True), (0.0, 0.5, False), (0.5, 0.5, False)],
)
def test_dropout_overrides_reach_dense_blocks(tiny_regression_data, input_rate, tree_rate, head_only):
    features, targets = tiny_regression_data
    model = NODERegressor(
        num_trees=8,
        depth=2,
        max_epochs=2,
        batch_size=24,
        input_dropout=0.0,
        tree_dropout=0.0,
        tree_dropout_only_head=head_only,
        device="cpu",
        verbose=0,
    ).fit(features, targets)

    with model._temporary_dropout_rates(input_dropout=input_rate, tree_dropout=tree_rate):
        assert model.module_.dense_block.input_dropout == input_rate
        assert model.module_.dense_block.tree_dropout == tree_rate

    result = model.predict_uncertainty(features[:5], num_samples=20, input_dropout=input_rate, tree_dropout=tree_rate)

    assert result["knowledge_uncertainty"].max() > 1e-6
    for component in (model, model.module_, model.module_.dense_block):
        assert component.input_dropout == 0.0
        assert component.tree_dropout == 0.0

    with pytest.raises(RuntimeError, match="prediction failed"):
        with model._temporary_dropout_rates(input_dropout=input_rate, tree_dropout=tree_rate):
            raise RuntimeError("prediction failed")
    for component in (model, model.module_, model.module_.dense_block):
        assert component.input_dropout == 0.0
        assert component.tree_dropout == 0.0


@pytest.mark.parametrize("experts", [[[0.99, 0.99], [0.99, 0.99]], [[0.9, 0.1], [0.1, 0.9]]])
def test_multilabel_uncertainty_uses_bernoulli_entropies(monkeypatch, experts):  #
    probabilities = np.repeat(np.asarray(experts, dtype=np.float64)[:, None, :], repeats=2, axis=1)
    model = NODEClassifier(
        num_trees=4,
        depth=2,
        criterion=torch.nn.BCEWithLogitsLoss,
        model_type="classification_multilabel",
        device="cpu",
        verbose=0,
    ).initialize()
    monkeypatch.setattr(model, "_mc_dropout_proba_samples", lambda features, num_samples: probabilities)
    # Deterministic output probabilities are distinct from the MC samples used for entropy.
    deterministic_probabilities = np.array([[0.75, 0.25], [0.4, 0.6]])
    monkeypatch.setattr(model, "predict_proba", lambda X: deterministic_probabilities)
    mean_probabilities = probabilities.mean(axis=0)  # Shape (n_datapoints, n_labels)

    result = model.predict_uncertainty(np.zeros((2, 3), dtype=np.float32), num_samples=2)

    expected_data = (
        -(probabilities * np.log(probabilities) + (1 - probabilities) * np.log1p(-probabilities))
        .sum(axis=-1)
        .mean(axis=0)
    )
    expected_total = -(
        mean_probabilities * np.log(mean_probabilities) + (1 - mean_probabilities) * np.log1p(-mean_probabilities)
    ).sum(axis=-1)
    labels = range(deterministic_probabilities.shape[1])
    expected_columns = (
        [f"target_{i}_pred" for i in labels]
        + [f"target_{i}_mean_predictions" for i in labels]
        + [f"proba_target_{i}" for i in labels]
        + ["knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]
    )
    assert list(result.columns) == expected_columns

    np.testing.assert_array_equal(
        result[[f"target_{i}_pred" for i in labels]].to_numpy(), deterministic_probabilities > 0.5
    )
    assert result[[f"target_{i}_mean_predictions" for i in labels]].isna().all().all()
    np.testing.assert_array_equal(result[[f"proba_target_{i}" for i in labels]], deterministic_probabilities)
    np.testing.assert_allclose(result["data_uncertainty"], expected_data)
    np.testing.assert_allclose(result["total_uncertainty"], expected_total)
    np.testing.assert_allclose(result["knowledge_uncertainty"], (expected_total - expected_data), atol=1e-12)

    def bernoulli_entropy(p):
        return -(p * np.log(p) + (1 - p) * np.log1p(-p))

    per_label_knowledge = np.maximum(
        bernoulli_entropy(mean_probabilities) - bernoulli_entropy(probabilities).mean(axis=0), 0.0
    )
    acquisition = model.predict_uncertainty(np.zeros((2, 3), dtype=np.float32), num_samples=2, uncertainty_for_opt=True)
    assert list(acquisition.columns) == ["knowledge_uncertainty"]
    np.testing.assert_allclose(acquisition["knowledge_uncertainty"], per_label_knowledge.max(axis=1), atol=1e-12)


@pytest.mark.parametrize("model_class", [NODERegressor, NODEClassifier])
def test_mc_dropout_restores_eval_after_error(tiny_regression_data, monkeypatch, model_class):
    features, targets = tiny_regression_data
    if model_class is NODEClassifier:
        targets = (targets > 0).astype(np.int64)
    model = model_class(
        num_trees=4,
        depth=2,
        max_epochs=1,
        input_dropout=0.1,
        batch_norm_continuous_input=True,
        device="cpu",
        verbose=0,
    ).fit(features, targets)

    def fail_forward(features):
        assert not model.module_.embedding_layer.cont_batch_norm.training
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model.module_, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="forward failed"):
        if model_class is NODEClassifier:
            model._mc_dropout_proba_samples(features, 3)
        else:
            model._predict_uncertainty_mc_dropout(features, num_samples=3)
    assert all(not module.training for module in model.module_.modules())


def test_flow_sample_vectors_preserve_joint_structure(monkeypatch):
    model = NODERegressor(num_trees=4, depth=2, device="cpu", verbose=0).initialize()
    samples = torch.tensor([[[3.0, -3.0]], [[1.0, -1.0]], [[2.0, -2.0]]])

    class FixedDistribution:
        def sample(self, shape):
            assert shape == torch.Size([3])
            return samples

    monkeypatch.setattr(model, "_prepare_data_for_node", lambda features: features, raising=False)
    monkeypatch.setattr(model, "forward_iter", lambda features, training: iter([FixedDistribution()]))

    result = model.predict_flow_head(np.zeros((1, 2), dtype=np.float32), num_samples=3, return_sample_distribution=True)

    np.testing.assert_array_equal(result, samples.permute(1, 0, 2).numpy())
    np.testing.assert_array_equal(result[..., 0] + result[..., 1], 0.0)


def test_flow_sampling_controls_are_forwarded(monkeypatch):
    model = NODERegressor(num_trees=4, depth=2, head_type="flow", device="cpu", verbose=0).initialize()
    calls = []

    def combined(features, **kwargs):
        calls.append(kwargs)
        return {
            "predictions": np.zeros(len(features)),
            "knowledge_uncertainty": np.ones(len(features)),
            "data_uncertainty": np.ones(len(features)),
            "total_uncertainty": np.full(len(features), 2.0),
        }

    monkeypatch.setattr(model, "predict_with_combined_uncertainty", combined)
    monkeypatch.setattr(model, "predict", lambda features, num_samples=1000: np.zeros(len(features)))
    model.predict_uncertainty(np.zeros((2, 1), dtype=np.float32), num_mc_samples=3, num_flow_samples=17)

    assert calls[0]["num_mc_samples"] == 3
    assert calls[0]["num_flow_samples"] == 17


@pytest.mark.parametrize("method", ["bald", "balsa_emd"])
@pytest.mark.parametrize("passes", [3, 5])
def test_flow_disagreement_evaluates_only_required_densities(monkeypatch, method, passes):
    model = NODERegressor(num_trees=4, depth=2, head_type="flow", device="cpu", verbose=0).initialize()
    calls = []
    samples = []
    distributions = []

    class RecordedNormal:
        def __init__(self, features):
            self.distribution = torch.distributions.Independent(
                torch.distributions.Normal(
                    torch.full((len(features), 1), 0.01 * len(distributions)),
                    torch.full((len(features), 1), 0.01),
                ),
                1,
            )
            distributions.append(self.distribution)

        def sample(self, shape):
            values = self.distribution.sample(shape)
            samples.append(values)
            return values

        def log_prob(self, values):
            calls.append(values.shape)
            return self.distribution.log_prob(values)

    monkeypatch.setattr(model, "_prepare_data_for_node", lambda features: features, raising=False)
    monkeypatch.setattr(model.module_, "forward", RecordedNormal)
    result = model.predict_with_combined_uncertainty(
        np.zeros((2, 1), dtype=np.float32),
        num_mc_samples=passes,
        num_flow_samples=40,
        knowledge_method=method,
        return_all=True,
    )

    assert len(calls) == (passes**2 if method == "bald" else passes)
    expected_data = (
        -torch.stack([distribution.log_prob(values) for distribution, values in zip(distributions, samples)])
        .mean(dim=(0, 1))
        .numpy()
    )
    np.testing.assert_allclose(result["data_uncertainty"], expected_data)
    np.testing.assert_allclose(result["predictions"], torch.stack(samples).mean(dim=(0, 1)).numpy().ravel())
    np.testing.assert_allclose(result["mc_stds"], torch.stack(samples).std(dim=1, unbiased=False).numpy())
    assert (result["data_uncertainty"] < 0).all()

    if method == "balsa_emd":
        sorted_samples = np.sort(torch.stack(samples).numpy()[..., 0], axis=1)
        expected_disagreement = np.abs(np.diff(sorted_samples, axis=0)).mean(axis=1).sum(axis=0)
        np.testing.assert_allclose(result["knowledge_uncertainty"], expected_disagreement, rtol=1e-6)
        assert np.isnan(result["total_uncertainty"]).all()
    else:
        mixture_log_probabilities = torch.stack(
            [
                torch.logsumexp(torch.stack([distribution.log_prob(values) for distribution in distributions]), dim=0)
                - np.log(passes)
                for values in samples
            ]
        )
        expected_total = -mixture_log_probabilities.mean(dim=(0, 1)).numpy()
        expected_knowledge = np.maximum(expected_total - expected_data, 0.0)
        np.testing.assert_allclose(result["knowledge_uncertainty"], expected_knowledge, atol=1e-6)
        np.testing.assert_allclose(result["total_uncertainty"], expected_data + expected_knowledge, atol=1e-6)


def test_regression_mean_predictions_is_mc_average(monkeypatch):
    model = NODERegressor(num_trees=4, depth=2, input_dropout=0.1, device="cpu", verbose=0).initialize()
    values = iter([0.0, 2.0])
    monkeypatch.setattr(model, "_prepare_data_for_node", lambda features: features, raising=False)
    monkeypatch.setattr(model, "predict", lambda features: np.full(len(features), 42.0))
    monkeypatch.setattr(model.module_, "forward", lambda features: torch.full((len(features), 1), next(values)))

    result = model.predict_uncertainty(np.zeros((2, 1), dtype=np.float32), num_samples=2)

    np.testing.assert_array_equal(result["pred"], [42.0, 42.0])
    np.testing.assert_array_equal(result["mean_predictions"], [1.0, 1.0])
    np.testing.assert_array_equal(result["knowledge_uncertainty"], [1.0, 1.0])


def test_embedding_dropout_is_recognized():
    features = pd.DataFrame({"category": ["one", "two", "three"] * 4})
    model = NODERegressor(
        cat_features=["category"],
        num_trees=8,
        depth=2,
        max_epochs=1,
        input_dropout=0.0,
        tree_dropout=0.0,
        embedding_dropout=0.5,
        device="cpu",
        verbose=0,
    ).fit(features, np.arange(12, dtype=np.float32))

    assert model._has_active_dropout()
    result = model.predict_uncertainty(features, num_samples=20)
    assert result["knowledge_uncertainty"].max() > 1e-6


def test_empty_mlp_does_not_report_nonexistent_dropout(tiny_regression_data):
    features, targets = tiny_regression_data
    model = NODERegressor(
        head_type="mlp",
        mlp_hidden_dims=[],
        mlp_dropout=0.3,
        input_dropout=0.0,
        tree_dropout=0.0,
        num_trees=4,
        depth=2,
        max_epochs=1,
        device="cpu",
        verbose=0,
    ).fit(features, targets)

    assert not model._has_active_dropout()
    with pytest.warns(UserWarning, match="MC-dropout repeats are deterministic"):
        result = model.predict_uncertainty(features, num_samples=3)
    np.testing.assert_array_equal(result["knowledge_uncertainty"], 0.0)


@pytest.mark.parametrize(
    ("model_class", "head_type", "mechanism"),
    [
        (NODERegressor, "subset", "input"),
        (NODERegressor, "linear", "internal_tree"),
        (NODERegressor, "mlp", "mlp"),
        (NODERegressor, "flow", "input"),
        (NODERegressor, "flow", "internal_tree"),
        (NODERegressor, "flow", "embedding"),
        (NODEClassifier, "subset", "head_tree"),
        (NODEClassifier, "subset", "embedding"),
        (NODEClassifier, "linear", "input"),
        (NODEClassifier, "mlp", "mlp"),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"))]
)
def test_isolated_mc_dropout_preserves_fitted_state(model_class, head_type, mechanism, device):
    if head_type == "flow":
        pytest.importorskip("zuko")
    rng = np.random.default_rng(42)
    features = pd.DataFrame({"measure": rng.normal(size=24).astype(np.float32), "category": ["a", "b", "c"] * 8})
    targets = features["measure"].to_numpy().copy()
    if model_class is NODEClassifier:
        targets = (targets > 0).astype(np.int64)
    model = model_class(
        head_type=head_type,
        num_trees=8,
        depth=2,
        num_layers=2,
        max_epochs=1,
        batch_size=8,
        input_dropout=0.3 if mechanism == "input" else 0.0,
        tree_dropout=0.3 if "tree" in mechanism else 0.0,
        tree_dropout_only_head=mechanism != "internal_tree",
        mlp_dropout=0.3 if mechanism == "mlp" else 0.0,
        mlp_hidden_dims=[8],
        cat_features=["category"],
        embedding_dropout=0.3 if mechanism == "embedding" else 0.0,
        batch_norm_continuous_input=True,
        device=device,
        verbose=0,
    ).fit(features, targets)
    test_features = features.iloc[:5]
    torch.manual_seed(43)
    before_prediction = model.predict(test_features)
    before_state = {name: value.clone() for name, value in model.module_.state_dict().items()}
    stochastic_outputs = []

    def capture_output(module, inputs, output):
        if module.training:
            for child in module.modules():
                if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                    assert not child.training
            if head_type == "flow":
                output = output.log_prob(torch.zeros(len(inputs[0]), 1, device=inputs[0].device))
            stochastic_outputs.append(output.detach().cpu())

    handle = model.module_.register_forward_hook(capture_output)
    try:
        result = model.predict_uncertainty(test_features, num_samples=8, num_flow_samples=12)
    finally:
        handle.remove()

    assert len(stochastic_outputs) == 8
    assert torch.stack(stochastic_outputs).var(dim=0, unbiased=False).max() > 1e-10
    assert np.isfinite(result["knowledge_uncertainty"]).all()
    assert all(not module.training for module in model.module_.modules())
    for name, value in model.module_.state_dict().items():
        torch.testing.assert_close(value, before_state[name], atol=0, rtol=0)
    torch.manual_seed(43)
    np.testing.assert_array_equal(model.predict(test_features), before_prediction)


@pytest.mark.parametrize("return_quantiles", [False, True])
def test_flow_dropout_overrides_restore_on_sampling_error(tiny_regression_data, monkeypatch, return_quantiles):
    pytest.importorskip("zuko")
    features, targets = tiny_regression_data
    model = NODERegressor(
        head_type="flow",
        num_trees=4,
        depth=2,
        max_epochs=1,
        input_dropout=0.0,
        tree_dropout=0.0,
        tree_dropout_only_head=False,
        batch_norm_continuous_input=True,
        device="cpu",
        verbose=0,
    ).fit(features, targets)

    class BrokenDistribution(torch.distributions.Distribution):
        arg_constraints = {}

        def __init__(self):
            super().__init__(batch_shape=torch.Size([3]), event_shape=torch.Size([1]))

        def sample(self, shape):
            assert model.module_.dense_block.input_dropout == 0.2
            assert model.module_.dense_block.tree_dropout == 0.3
            assert not model.module_.embedding_layer.cont_batch_norm.training
            raise RuntimeError("sampling failed")

    monkeypatch.setattr(model.module_, "forward", lambda features: BrokenDistribution())
    with pytest.raises(RuntimeError, match="sampling failed"):
        model.predict_uncertainty(
            features[:3],
            input_dropout=0.2,
            tree_dropout=0.3,
            return_quantiles=return_quantiles,
            num_samples=4,
            num_mc_samples=4,
            num_flow_samples=8,
        )
    assert all(not module.training for module in model.module_.modules())
    for component in (model, model.module_, model.module_.dense_block):
        assert component.input_dropout == 0.0
        assert component.tree_dropout == 0.0


@pytest.mark.filterwarnings("ignore:You set iterator_valid__shuffle=True:UserWarning")
@pytest.mark.parametrize("model_class", [NODERegressor, NODEClassifier])
@pytest.mark.parametrize("invalid_iterator", ["shuffle", "drop_last", "sampler"])
def test_mc_dropout_rejects_misaligned_inference_rows(tiny_regression_data, model_class, invalid_iterator):
    features, targets = tiny_regression_data
    if model_class is NODEClassifier:
        targets = (targets > 0).astype(np.int64)
    model = model_class(num_trees=4, depth=2, max_epochs=1, batch_size=8, device="cpu", verbose=0).fit(
        features, targets
    )
    generator = torch.Generator().manual_seed(0)
    option = (
        torch.utils.data.RandomSampler(range(len(features)), generator=generator)
        if invalid_iterator == "sampler"
        else True
    )
    model.set_params(**{f"iterator_valid__{invalid_iterator}": option, "iterator_valid__generator": generator})

    with pytest.raises(ValueError, match="NODE inference requires"):
        model.predict_uncertainty(features[:5], num_samples=3)
    assert all(not module.training for module in model.module_.modules())


@pytest.mark.filterwarnings("ignore:You set iterator_valid__shuffle=True:UserWarning")
@pytest.mark.parametrize("return_quantiles", [False, True])
def test_flow_mc_dropout_rejects_shuffled_rows(tiny_regression_data, return_quantiles):
    pytest.importorskip("zuko")
    features, targets = tiny_regression_data
    model = NODERegressor(
        head_type="flow",
        num_trees=4,
        depth=2,
        max_epochs=1,
        iterator_valid__shuffle=True,
        iterator_valid__generator=torch.Generator().manual_seed(0),
        device="cpu",
        verbose=0,
    ).fit(features, targets)

    with pytest.raises(ValueError, match="input order"):
        model.predict_uncertainty(
            features[:5],
            num_samples=3,
            num_mc_samples=3,
            num_flow_samples=5,
            return_quantiles=return_quantiles,
        )
    assert all(not module.training for module in model.module_.modules())


def test_sequential_prediction_batches_preserve_rows(tiny_regression_data):
    features, targets = tiny_regression_data
    model = NODERegressor(num_trees=4, depth=2, max_epochs=1, batch_size=8, device="cpu", verbose=0).fit(
        features, targets
    )
    for _ in range(2):
        rows = torch.cat(list(model.get_iterator(features[:19], training=False))).numpy()
        np.testing.assert_array_equal(rows, features[:19])
    assert isinstance(model.get_iterator(features, training=True).sampler, torch.utils.data.RandomSampler)


@pytest.mark.parametrize("sampling", ["sampler", "batch_sampler_subclass", "variable_batches"])
def test_inference_accepts_ordered_custom_sampling(tiny_regression_data, sampling):
    features, targets = tiny_regression_data
    inputs = features[:19]

    class OrderedBatches(torch.utils.data.BatchSampler):
        pass

    if sampling == "sampler":
        iterator_params = {"iterator_valid__sampler": list(range(len(inputs)))}
    else:
        batches = (
            OrderedBatches(range(len(inputs)), batch_size=8, drop_last=False)
            if sampling == "batch_sampler_subclass"
            else [[0], list(range(1, 7)), list(range(7, len(inputs)))]
        )
        iterator_params = {"iterator_valid__batch_size": 1, "iterator_valid__batch_sampler": batches}

    model = NODERegressor(
        num_trees=4, depth=2, max_epochs=1, batch_size=8, device="cpu", verbose=0, **iterator_params
    ).fit(features, targets)
    model.module_.eval()
    with torch.no_grad():
        expected = model._to_target_units(model.module_(torch.from_numpy(inputs)).numpy())

    np.testing.assert_allclose(model.predict(inputs), expected.ravel(), rtol=1e-5, atol=1e-6)


def test_inference_rejects_reordered_batch_sampler(tiny_regression_data):
    features, targets = tiny_regression_data
    inputs = features[:19]
    batches = torch.utils.data.BatchSampler(list(reversed(range(len(inputs)))), batch_size=8, drop_last=False)
    model = NODERegressor(
        num_trees=4,
        depth=2,
        max_epochs=1,
        batch_size=8,
        device="cpu",
        verbose=0,
        iterator_valid__batch_size=1,
        iterator_valid__batch_sampler=batches,
    ).fit(features, targets)

    with pytest.raises(ValueError, match="input order"):
        model.predict(inputs)


@pytest.mark.filterwarnings("ignore:You set iterator_valid__shuffle=True:UserWarning")
@pytest.mark.parametrize("model_class", [NODERegressor, NODEClassifier])
@pytest.mark.parametrize("option", ["shuffle", "drop_last"])
def test_fit_validation_allows_shuffled_rows(tiny_regression_data, model_class, option):
    from skorch.dataset import ValidSplit

    features, targets = tiny_regression_data
    if model_class is NODEClassifier:
        targets = (targets > 0).astype(np.int64)
    model = model_class(
        num_trees=4,
        depth=2,
        max_epochs=1,
        batch_size=7,
        device="cpu",
        verbose=0,
        train_split=ValidSplit(cv=2),
        **{f"iterator_valid__{option}": True},
    ).fit(features, targets)

    assert np.isfinite(model.history[-1, "valid_loss"])


@pytest.mark.parametrize(
    "batches",
    [
        [[0, 1], [3, 2]],
        [[0, 1], [1, 3]],
        [[0, 1], [3]],
        [[0, 1]],
        [[0, 1, 2, 3], [0]],
        [[-1, 0, 1, 2]],
        [[0, 1, 2, 4]],
        [[0.0, 1.0, 2.0, 3.0]],
        [[]],
        [],
    ],
    ids=[
        "reordered",
        "duplicate",
        "skipped",
        "truncated",
        "extra",
        "negative",
        "out-of-range",
        "float",
        "empty-batch",
        "empty",
    ],
)
def test_inference_checks_actual_row_ids(batches):
    features = np.ones((4, 2), dtype=np.float32)
    model = NODERegressor(device="cpu", iterator_valid__batch_size=1, iterator_valid__batch_sampler=batches)
    with pytest.raises(ValueError, match="every input row exactly once in input order"):
        list(model._get_inference_iterator(features))


@pytest.mark.parametrize(
    "iterator_params",
    [
        {"iterator_valid__batch_size": None},
        {"iterator_valid__batch_size": -1},
        {"iterator_valid__batch_size": 2, "iterator_valid__drop_last": True},
        {"iterator_valid__num_workers": 2, "iterator_valid__multiprocessing_context": "spawn"},
    ],
    ids=["unbatched", "full-batch", "drop-last-without-omission", "worker-processes"],
)
def test_inference_keeps_safe_loader_options(iterator_params):
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    model = NODERegressor(batch_size=2, device="cpu", **iterator_params)
    rows = torch.cat(list(model._get_inference_iterator(features))).numpy()
    np.testing.assert_array_equal(rows, features)


@pytest.mark.parametrize("reverse_batches", [False, True])
def test_inference_checks_custom_iterator_delivery(reverse_batches):
    class CustomIterator:
        def __init__(self, dataset, batch_size, collate_fn):
            self.batches = [
                collate_fn([dataset[row] for row in range(start, min(start + batch_size, len(dataset)))])
                for start in range(0, len(dataset), batch_size)
            ]

        def __iter__(self):
            return iter(self.batches[::-1] if reverse_batches else self.batches)

    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    model = NODERegressor(batch_size=2, iterator_valid=CustomIterator, device="cpu")
    if reverse_batches:
        with pytest.raises(ValueError, match="input order"):
            list(model._get_inference_iterator(features))
    else:
        rows = torch.cat(list(model._get_inference_iterator(features))).numpy()
        np.testing.assert_array_equal(rows, features)


def test_inference_does_not_preconsume_sampler():
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    model = NODERegressor(batch_size=2, iterator_valid__sampler=iter(range(4)), device="cpu")
    rows = torch.cat(list(model._get_inference_iterator(features))).numpy()
    np.testing.assert_array_equal(rows, features)
    with pytest.raises(ValueError, match="input order"):
        list(model._get_inference_iterator(features))


@pytest.mark.parametrize("model_class", [NODERegressor, NODEClassifier])
def test_inference_iterator_can_be_reset_without_refitting(tiny_regression_data, model_class):
    features, targets = tiny_regression_data
    if model_class is NODEClassifier:
        targets = (targets > 0).astype(np.int64)
    model = model_class(
        num_trees=4,
        depth=2,
        max_epochs=1,
        batch_size=8,
        device="cpu",
        verbose=0,
        iterator_valid__drop_last=True,
    ).fit(features, targets)
    fitted_module = model.module_
    fitted_optimizer = model.optimizer_
    fitted_state = {name: value.clone() for name, value in model.module_.state_dict().items()}
    with pytest.raises(ValueError, match="iterator_valid__drop_last=False"):
        model.predict(features[:5])

    returned = model.set_params(
        iterator_valid__shuffle=False,
        iterator_valid__drop_last=False,
        iterator_valid__sampler=None,
        iterator_valid__batch_sampler=None,
    )
    assert returned is model
    assert model.module_ is fitted_module
    assert model.optimizer_ is fitted_optimizer
    assert len(model.predict(features[:5])) == 5
    for name, value in model.module_.state_dict().items():
        torch.testing.assert_close(value, fitted_state[name], atol=0, rtol=0)


@pytest.mark.parametrize("task", ["regression", "classification", "flow"])
def test_mc_passes_keep_rows_aligned_with_changing_batch_sizes(tiny_regression_data, monkeypatch, task):  #
    if task == "flow":
        pytest.importorskip("zuko")
    features, targets = tiny_regression_data
    inputs = features[:19]

    class ChangingBatches:
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            batch_size = 3 if self.iterations % 2 else 7
            for start in range(0, len(inputs), batch_size):
                yield list(range(start, min(start + batch_size, len(inputs))))

    batches = ChangingBatches()
    model_class = NODEClassifier if task == "classification" else NODERegressor
    if task == "classification":
        targets = (targets > 0).astype(np.int64)
    model = model_class(
        head_type="flow" if task == "flow" else "subset",
        num_trees=4,
        depth=2,
        max_epochs=1,
        batch_size=8,
        device="cpu",
        verbose=0,
        input_dropout=0.2,
        iterator_valid__batch_size=1,
        iterator_valid__batch_sampler=batches,
    ).fit(features, targets)

    def deterministic_forward(batch):
        location = batch[:, :1]
        if task == "classification":
            return torch.cat([location, -location], dim=1)
        if task == "flow":
            location = (location - model.target_scaler_.mean_[0]) / model.target_scaler_.scale_[0]
            return torch.distributions.Independent(torch.distributions.Normal(location, 1e-5), 1)
        # Regression modules emit standardized targets; predictions map them back to original units.
        return (location - model.target_scaler_.mean_[0]) / model.target_scaler_.scale_[0]

    monkeypatch.setattr(model.module_, "forward", deterministic_forward)
    result = model.predict_uncertainty(
        inputs, num_samples=3, num_mc_samples=3, num_flow_samples=8, return_quantiles=task == "flow"
    )
    if task == "flow":
        result, quantiles = result
        np.testing.assert_allclose(quantiles[:, 1], inputs[:, 0], atol=1e-4)
    expected = (
        torch.softmax(deterministic_forward(torch.from_numpy(inputs)), dim=1).numpy()[:, 1]
        if task == "classification"
        else inputs[:, 0]
    )
    # Classifiers leave mean_predictions unset; the probability is in proba_1.
    column = "proba_1" if task == "classification" else "mean_predictions"
    np.testing.assert_allclose(result[column], expected, atol=1e-4)
    if task == "flow":
        np.testing.assert_allclose(result["pred"], inputs[:, 0], atol=1e-4)
    # Flow heads read the data for quantiles, the MC-dropout passes and the deterministic mode.
    # Classifiers now require two passes: one for MC-dropout samples and one for deterministic predict_proba.
    assert batches.iterations == {"classification": 2, "regression": 2, "flow": 3}[task]
    assert all(not module.training for module in model.module_.modules())


@pytest.mark.parametrize("count", [0, -1])
def test_classifier_mc_dropout_requires_positive_sample_count(tiny_regression_data, count):
    features, targets = tiny_regression_data
    model = NODEClassifier(num_trees=4, depth=2, max_epochs=1, device="cpu", verbose=0).fit(
        features, (targets > 0).astype(int)
    )
    with pytest.raises(ValueError, match="num_samples must be positive"):
        model.predict_uncertainty(features[:5], num_samples=count)


def _cv_frame():
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"), index=[f"m{i}" for i in range(60)])
    return X, KFold(2, shuffle=True, random_state=0)


_NODE_CV_PARAMS = dict(num_trees=4, depth=2, max_epochs=2, input_dropout=0.1, device="cpu", verbose=0)


def test_binary_classifier_uncertainty_layout_in_mother_cv():
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series((X["a"] > 0).astype(int), name="y")

    node = mother_cv(NODEClassifier(**_NODE_CV_PARAMS), X=X, y=y, cv=cv)

    node_columns = list(node.columns)
    assert node_columns[:6] == [
        "pred_y",
        "pred_proba_0",
        "pred_proba_1",
        "pred_knowledge_uncertainty",
        "pred_data_uncertainty",
        "pred_total_uncertainty",
    ]
    assert "pred_mean_predictions" not in node_columns  # unset for classifiers, so mother_cv drops it
    np.testing.assert_allclose(node[["pred_proba_0", "pred_proba_1"]].sum(axis=1), 1.0, atol=1e-5)
    assert (node["pred_y"] == (node["pred_proba_1"] > 0.5).astype(int)).all()


def test_multiclass_classifier_proba_columns_in_mother_cv():
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series(pd.cut(X["a"], [-9, -0.5, 0.5, 9], labels=[0, 1, 2]).astype(int), name="y")

    result = mother_cv(NODEClassifier(**_NODE_CV_PARAMS), X=X, y=y, cv=cv)

    assert list(result.columns[:4]) == ["pred_y", "pred_proba_0", "pred_proba_1", "pred_proba_2"]
    np.testing.assert_allclose(result[["pred_proba_0", "pred_proba_1", "pred_proba_2"]].sum(axis=1), 1.0, atol=1e-5)
    assert (
        result["pred_y"] == result[["pred_proba_0", "pred_proba_1", "pred_proba_2"]].to_numpy().argmax(axis=1)
    ).all()


def test_multilabel_and_multitarget_outputs_work_in_mother_cv():
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    labels = pd.DataFrame({"l0": (X["a"] > 0).astype(int), "l1": (X["b"] > 0).astype(int)})
    classifier = NODEClassifier(
        criterion=torch.nn.BCEWithLogitsLoss, model_type="classification_multilabel", **_NODE_CV_PARAMS
    )
    result = mother_cv(classifier, X=X, y=labels, cv=cv, num_samples=10)
    assert {"pred_target_0_pred", "pred_target_1_pred", "pred_knowledge_uncertainty"} <= set(result.columns)
    assert not any(result[f"pred_target_{i}_pred"].map(lambda v: hasattr(v, "__len__")).any() for i in range(2))
    assert {"pred_proba_target_0", "pred_proba_target_1"} <= set(result.columns)

    targets = pd.DataFrame({"t0": X["a"] * 10 + 100, "t1": X["b"] * 0.1})
    for params in ({"head_type": "linear"}, {"head_type": "flow", "flow_type": "NICE"}):
        # Flow heads with dropout scale quadratically with num_samples (mixture entropy over MC passes).
        regression = mother_cv(NODERegressor(**params, **_NODE_CV_PARAMS), X=X, y=targets, cv=cv, num_samples=10)
        assert {"pred_target_0_pred", "pred_target_1_pred", "pred_target_0_mean_predictions"} <= set(regression.columns)
        assert np.isfinite(regression[["pred_target_0_pred", "pred_target_1_pred"]].to_numpy(dtype=float)).all()


@pytest.mark.parametrize(
    "negative, positive",
    [("safe", "toxic"), (5, 7)],
    ids=["string_labels", "non_contiguous_integer_labels"],
)
def test_classifier_probability_columns_are_positional_in_mother_cv(negative, positive):
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series(np.where(X["a"] > 0, positive, negative), index=X.index, name="y")
    model = NODEClassifier(**_NODE_CV_PARAMS).fit(X, y)

    direct = model.predict_uncertainty(X.iloc[:5], num_samples=5)
    ordered = np.asarray(model.classes_)  # sorted labels: the class order of proba_0, proba_1, ...
    assert list(direct.columns[:3]) == ["pred", "proba_0", "proba_1"]
    assert direct["mean_predictions"].isna().all()  # unset for classifiers; probabilities are in proba_*
    np.testing.assert_allclose(direct[["proba_0", "proba_1"]].sum(axis=1), 1.0, atol=1e-5)

    result = mother_cv(NODEClassifier(**_NODE_CV_PARAMS), X=X, y=y, cv=cv)
    assert list(result.columns[:3]) == ["pred_y", "pred_proba_0", "pred_proba_1"]
    assert set(result["pred_y"]) <= {negative, positive}
    probabilities = result[["pred_proba_0", "pred_proba_1"]].to_numpy()
    assert (result["pred_y"].to_numpy() == ordered[probabilities.argmax(axis=1)]).all()


# Tiny NODE settings for nested CV (outer folds × Optuna trials × inner folds).
_NODE_NESTED_PARAMS = dict(num_trees=4, depth=2, max_epochs=1, device="cpu", verbose=0)


def test_single_target_regressor_uncertainty_layout_in_mother_cv():
    """Plain single-target NODERegressor through mother_cv (multi-target is covered above)."""
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series(X["a"] * 10.0 + 100.0, index=X.index, name="y")

    result = mother_cv(NODERegressor(**_NODE_CV_PARAMS), X=X, y=y, cv=cv)

    assert result.shape[0] == len(X)
    # Target series is named "y", so mother_cv prefixes the point prediction as pred_y.
    assert {"pred_y", "pred_mean_predictions", "pred_total_uncertainty", "iteration"} <= set(result.columns)
    assert len(result["iteration"].unique()) == 2


@pytest.mark.parametrize(
    "model_cls, y_from_X, scorer, expected_pred_col",
    [
        (NODEClassifier, lambda X: pd.Series((X["a"] > 0).astype(int), index=X.index, name="y"), "accuracy", "pred_y"),
        (
            NODERegressor,
            lambda X: pd.Series(X["a"] * 10.0 + 100.0, index=X.index, name="y"),
            "neg_mean_squared_error",
            "pred_y",
        ),
    ],
    ids=["classification", "regression"],
)
def test_mother_cv_with_tuner_runs_nested_optimization(model_cls, y_from_X, scorer, expected_pred_col):
    """Minimal mother_cv + MotherTuner path for NODE (excluded from the generic algorithm sweeps)."""
    from mother.optimization import MotherTuner
    from mother.pipeline_utils import mother_cv

    def tiny_space(X, y, trial, prefix=""):
        # Keep nested CV fast: only tune lr; architecture stays at _NODE_NESTED_PARAMS.
        return {f"{prefix}lr": trial.suggest_float(f"{prefix}lr", 1e-3, 1e-2, log=True)}

    X, cv = _cv_frame()
    y = y_from_X(X)
    tuner = MotherTuner(scorer=scorer, n_trials_optuna=2, n_threads_optuna=1, n_startup_trials=1)

    result = mother_cv(
        model_cls(**_NODE_NESTED_PARAMS),
        X=X,
        y=y,
        cv=cv,
        inner_cv=cv,
        tuner=tuner,
        hyperparameter_space_function=tiny_space,
        default_parameters={"lr": 5e-3},
    )

    assert result.shape[0] == len(X)
    assert expected_pred_col in result.columns
    assert len(result["iteration"].unique()) == 2
    assert tuner.study is not None
    assert len(tuner.study.trials) >= 1


class _FastArchitectureTrial:
    """Proxy Optuna trial: runs the real NODE suggest graph, but clamps costly knobs.

    ``get_hyperparameter_space`` still calls every ``suggest_*`` (heads, dropout,
    activations, flow params, …); only tree budget / depth / flow width are forced
    into a tiny range so nested CV stays fast.
    """

    def __init__(self, trial):
        self._trial = trial

    def suggest_int(self, name, low, high, *args, **kwargs):
        key = name.rsplit("__", 1)[-1]
        if key == "total_trees":
            return self._trial.suggest_int(name, 4, 8, step=4)
        if key == "num_layers":
            return self._trial.suggest_int(name, 1, 1)
        if key == "depth":
            return self._trial.suggest_int(name, 2, 2)
        if key == "additional_tree_output_dim":
            return self._trial.suggest_int(name, 0, 3)
        if key in {"flow_transforms", "flow_bins", "flow_components"}:
            return self._trial.suggest_int(name, 2, 2)
        if key in {"flow_signal", "flow_degree"}:
            return self._trial.suggest_int(name, 8, 8)
        if key.startswith("mlp_"):
            return self._trial.suggest_int(name, low, min(high, max(low, 32)), *args, **kwargs)
        return self._trial.suggest_int(name, low, high, *args, **kwargs)

    def suggest_float(self, *args, **kwargs):
        return self._trial.suggest_float(*args, **kwargs)

    def suggest_categorical(self, name, choices, *args, **kwargs):
        key = name.rsplit("__", 1)[-1]
        # Keep flow nested CV on the lightest coupling architecture.
        if key == "flow_type" and "NICE" in choices:
            choices = ("NICE",)
        if key == "head_type" and "linear" in choices:
            # Avoid sampling mlp/flow here when exercising the generic regressor space.
            choices = tuple(c for c in choices if c in ("linear", "subset")) or choices
        return self._trial.suggest_categorical(name, choices, *args, **kwargs)

    def set_user_attr(self, *args, **kwargs):
        return self._trial.set_user_attr(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._trial, name)


def _clamped_node_space(model):
    def space(X, y, trial, prefix=""):
        return model.get_hyperparameter_space(X, y, _FastArchitectureTrial(trial), prefix=prefix)

    return space


def _tiny_node_defaults(model, prefix=""):
    defaults = model.default_parameters(prefix=prefix)
    defaults[prefix + "total_trees"] = 4
    defaults[prefix + "depth"] = 2
    defaults[prefix + "num_layers"] = 1
    defaults[prefix + "batch_size"] = 32
    if prefix + "flow_transforms" in defaults or getattr(model, "head_type", None) == "flow":
        defaults[prefix + "flow_type"] = "NICE"
        defaults[prefix + "flow_transforms"] = 2
    return defaults


def test_mother_cv_return_estimators_for_node():
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series((X["a"] > 0).astype(int), index=X.index, name="y")

    table, estimators = mother_cv(
        NODEClassifier(**_NODE_NESTED_PARAMS),
        X=X,
        y=y,
        cv=cv,
        return_estimators=True,
    )

    assert table.shape[0] == len(X)
    assert "estimators" in estimators
    assert len(estimators["estimators"]) == cv.get_n_splits()
    assert estimators["prediction_prefix"] == "pred_"
    assert estimators["target_columns"] == ["y"]


def test_mother_cv_grouped_splits_for_node():
    from sklearn.model_selection import GroupKFold

    from mother.pipeline_utils import mother_cv

    X, _ = _cv_frame()
    y = pd.Series(X["a"] * 10.0 + 100.0, index=X.index, name="y")
    groups = pd.DataFrame({"g": np.tile(np.arange(5), len(X) // 5 + 1)[: len(X)]}, index=X.index)
    cv = GroupKFold(n_splits=2)

    result = mother_cv(NODERegressor(**_NODE_NESTED_PARAMS), X=X, y=y, cv=cv, groups=groups)

    assert result.shape[0] == len(X)
    assert "cv_group" in result.columns
    assert result["cv_group"].notna().all()
    assert set(result["iteration"]) == {0, 1}


def test_mother_cv_with_clamped_full_hyperparameter_space():
    """Exercise the real NODE suggest graph under mother_cv+tuner, with architecture clamped for speed."""
    from mother.optimization import MotherTuner
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series(X["a"] * 10.0 + 100.0, index=X.index, name="y")
    model = NODERegressor(**_NODE_NESTED_PARAMS, batch_size=32, tune_head=True)
    tuner = MotherTuner(scorer="neg_mean_squared_error", n_trials_optuna=2, n_threads_optuna=1, n_startup_trials=1)

    result = mother_cv(
        model,
        X=X,
        y=y,
        cv=cv,
        inner_cv=cv,
        tuner=tuner,
        hyperparameter_space_function=_clamped_node_space(model),
        default_parameters=_tiny_node_defaults(model),
    )

    assert result.shape[0] == len(X)
    assert "pred_y" in result.columns
    assert tuner.study is not None
    # Real space keys (not just lr) were registered on the study.
    sampled_keys = set().union(*(t.params.keys() for t in tuner.study.trials))
    assert {"total_trees", "num_layers", "depth", "lr", "head_type"} <= sampled_keys


def test_mother_cv_pipeline_with_feature_selection_and_tuner():
    from mother.ml import PipelineWithHyperparameterRooting
    from mother.optimization import MotherTuner
    from mother.pipeline_utils import get_feature_selection_pipeline, mother_cv

    X, cv = _cv_frame()
    y = pd.Series((X["a"] > 0).astype(int), index=X.index, name="y")
    feature_selector = get_feature_selection_pipeline(
        settings={
            "feature_selection_flags": ["DROP_CORRELATED"],
            "correlation_threshold": 0.99,
            "algorithm": "node",
            "model_type": "classification_binary",
            "target_type": "single_target",
        },
        pipeline_settings={"remainder": "passthrough", "verbose_feature_names_out": False},
    ).set_output(transform="pandas")
    pipeline = PipelineWithHyperparameterRooting(
        [("features", feature_selector), ("model", NODEClassifier(**_NODE_NESTED_PARAMS, batch_size=32))]
    )

    def tiny_space(X, y, trial, prefix=""):
        return {f"{prefix}model__lr": trial.suggest_float(f"{prefix}model__lr", 1e-3, 1e-2, log=True)}

    tuner = MotherTuner(scorer="accuracy", n_trials_optuna=2, n_threads_optuna=1, n_startup_trials=1)
    result = mother_cv(
        pipeline,
        X=X,
        y=y,
        cv=cv,
        inner_cv=cv,
        tuner=tuner,
        hyperparameter_space_function=tiny_space,
        default_parameters={"model__lr": 5e-3},
    )

    assert result.shape[0] == len(X)
    assert "pred_y" in result.columns
    assert tuner.study is not None


def test_mother_cv_flow_head_with_tuner():
    """Flow head is opt-in (tune_head=False); nested CV tunes flow knobs with a clamped budget."""
    pytest.importorskip("zuko")
    from mother.optimization import MotherTuner
    from mother.pipeline_utils import mother_cv

    X, cv = _cv_frame()
    y = pd.Series(X["a"] * 10.0 + 100.0, index=X.index, name="y")
    model = NODERegressor(
        **_NODE_NESTED_PARAMS,
        batch_size=32,
        head_type="flow",
        flow_type="NICE",
        tune_head=False,
        input_dropout=0.05,
    )
    tuner = MotherTuner(scorer="neg_mean_squared_error", n_trials_optuna=2, n_threads_optuna=1, n_startup_trials=1)

    result = mother_cv(
        model,
        X=X,
        y=y,
        cv=cv,
        inner_cv=cv,
        tuner=tuner,
        hyperparameter_space_function=_clamped_node_space(model),
        default_parameters=_tiny_node_defaults(model),
    )

    assert result.shape[0] == len(X)
    assert "pred_y" in result.columns
    sampled_keys = set().union(*(t.params.keys() for t in tuner.study.trials))
    assert "flow_type" in sampled_keys or "flow_transforms" in sampled_keys
