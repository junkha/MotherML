import numpy as np
import pandas as pd
import pytest

from mother.calibration.probability_calibration import (
    PlattProbabilityCalibrator,
    VennAbersProbabilityCalibrator,
)


@pytest.fixture
def probability_calibration_data():
    rng = np.random.default_rng(42)
    n_samples = 100
    proba_1 = np.linspace(0.01, 0.99, n_samples)
    probability_of_class_1 = 1 / (1 + np.exp(-(-1.5 + 2.5 * proba_1)))
    endog = rng.binomial(1, probability_of_class_1)
    exog = pd.DataFrame({"proba_0": 1 - proba_1, "proba_1": proba_1})
    return exog, endog


def _assert_valid_probabilities(probabilities, n_samples):
    probabilities = np.asarray(probabilities)
    assert probabilities.shape == (n_samples, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_platt_calibrator_returns_two_class_probabilities(probability_calibration_data):
    exog, endog = probability_calibration_data
    calibrator = PlattProbabilityCalibrator()

    assert calibrator.fit(exog=exog, endog=endog) is calibrator
    _assert_valid_probabilities(calibrator.predict_proba(exog=exog), len(exog))


def test_venn_abers_calibrator_returns_two_class_probabilities(probability_calibration_data):
    exog, endog = probability_calibration_data
    calibrator = VennAbersProbabilityCalibrator()

    assert calibrator.fit(exog=exog, endog=endog) is calibrator
    _assert_valid_probabilities(calibrator.predict_proba(exog=exog), len(exog))
