import numpy as np
import pandas as pd
import pytest

from mother.calibration.uncertainty_calibration import (
    BaggingIRErrorCalibrator,
    GLMErrorCalibrator,
    IsotonicRegressionErrorCalibrator,
    RandomForestErrorCalibrator,
)


@pytest.fixture
def uncertainty_calibration_data():
    rng = np.random.default_rng(42)
    n_samples = 100
    knowledge = np.linspace(0.01, 0.99, n_samples)
    data = rng.uniform(0.01, 0.99, n_samples)
    probability_of_error = 1 / (1 + np.exp(-(-1.5 + 2.5 * knowledge + 0.8 * data)))
    endog = rng.binomial(1, probability_of_error)

    exog = pd.DataFrame(
        {
            "knowledge_uncertainty": knowledge,
            "data_uncertainty": data,
            "total_uncertainty": knowledge + data,
        }
    )
    return exog, endog


@pytest.mark.parametrize(
    "calibrator_type",
    [BaggingIRErrorCalibrator, RandomForestErrorCalibrator],
)
def test_ensemble_error_calibrators_fit_and_predict(uncertainty_calibration_data, calibrator_type):
    exog, endog = uncertainty_calibration_data
    calibrator = (
        BaggingIRErrorCalibrator(n_estimators=30)
        if calibrator_type is BaggingIRErrorCalibrator
        else RandomForestErrorCalibrator(n_estimators=30)
    )

    assert calibrator.fit(exog=exog, endog=endog) is calibrator
    predictions = calibrator.predict(exog=exog)

    assert predictions.shape == (len(exog),)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
    assert np.isfinite(calibrator.oob_score)


def test_glm_error_calibrator_predicts_error_probabilities(uncertainty_calibration_data):
    exog, endog = uncertainty_calibration_data
    exog = exog[["knowledge_uncertainty", "data_uncertainty"]]
    calibrator = GLMErrorCalibrator()

    assert calibrator.fit(exog=exog, endog=endog) is calibrator
    predictions = calibrator.predict(exog=exog)

    assert predictions.shape == (len(exog),)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_isotonic_error_calibrator_is_monotonic_and_uses_configured_feature(uncertainty_calibration_data):
    exog, endog = uncertainty_calibration_data
    calibrator = IsotonicRegressionErrorCalibrator(uncertainty_type="data_uncertainty")

    assert calibrator.fit(exog=exog, endog=endog) is calibrator
    predictions = calibrator.predict(exog=exog)

    assert predictions.shape == (len(exog),)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()

    ordered = np.argsort(exog["data_uncertainty"].to_numpy())
    assert np.all(np.diff(predictions[ordered]) >= -1e-12)
