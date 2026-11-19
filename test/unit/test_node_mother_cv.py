"""Fast NODE-to-mother_cv probability-output integration tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

pytest.importorskip("skorch")
torch = pytest.importorskip("torch")

from mother.ml.models.m_node import NODEClassifier  # noqa: E402
from mother.pipeline_utils import mother_cv  # noqa: E402


def test_multilabel_probability_columns_are_integrated_by_mother_cv(monkeypatch):
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.DataFrame(
        {
            "label_a": [0, 0, 1, 1],
            "label_b": [1, 1, 0, 0],
        },
        index=X.index,
    )
    model = NODEClassifier(
        criterion=torch.nn.BCEWithLogitsLoss,
        model_type="classification_multilabel",
        num_trees=2,
        depth=1,
        max_epochs=1,
        device="cpu",
        verbose=0,
    )

    def fake_fit(self, X, y):
        return self

    def fake_predict_uncertainty(self, X, **kwargs):
        proba_0 = (X["feature"].to_numpy() + 1) / 5
        proba_1 = 1 - proba_0
        return pd.DataFrame(
            {
                "target_0_pred": (proba_0 > 0.5).astype(int),
                "target_1_pred": (proba_1 > 0.5).astype(int),
                "target_0_mean_predictions": None,
                "target_1_mean_predictions": None,
                "proba_target_0": proba_0,
                "proba_target_1": proba_1,
                "knowledge_uncertainty": np.zeros(len(X)),
                "data_uncertainty": np.zeros(len(X)),
                "total_uncertainty": np.zeros(len(X)),
            },
            index=X.index,
        )

    def unexpected_predict_proba(self, X):
        raise AssertionError("mother_cv should reuse predict_uncertainty probability columns")

    monkeypatch.setattr(NODEClassifier, "fit", fake_fit)
    monkeypatch.setattr(NODEClassifier, "predict_uncertainty", fake_predict_uncertainty)
    monkeypatch.setattr(NODEClassifier, "predict_proba", unexpected_predict_proba)

    result = mother_cv(model, X=X, y=y, cv=KFold(n_splits=2))

    assert {"pred_target_0_pred", "pred_target_1_pred", "pred_proba_target_0", "pred_proba_target_1"} <= set(
        result.columns
    )
    assert not {"pred_target_0_mean_predictions", "pred_target_1_mean_predictions"} & set(result.columns)
    np.testing.assert_allclose(result["pred_proba_target_0"], (X["feature"] + 1) / 5)
    np.testing.assert_allclose(result["pred_proba_target_1"], 1 - (X["feature"] + 1) / 5)
