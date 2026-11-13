from abc import abstractmethod
from typing import Protocol, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Calibrator(Protocol):
    """Base protocol for all calibrators.

    A Calibrator implements a fit-predict interface similar to scikit-learn estimators.
    It transforms raw model outputs into calibrated estimates.
    """

    @abstractmethod
    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the calibrator to calibration data.

        Args:
            exog: DataFrame containing raw model outputs (features) to calibrate.
            endog: True labels or outcomes corresponding to the samples in exog.

        Returns:
            self: The fitted calibrator object.
        """


class UncertaintyCalibrator(Calibrator, Protocol):
    """Protocol for calibrators that estimate model prediction uncertainty.

    Uncertainty calibrators transform raw uncertainty estimates (e.g., model confidence,
    ensemble variance) into reliable uncertainty quantification that reflects the
    true probability of model error.

    Example:
        An uncertainty calibrator might transform model confidence scores into
        estimates of the probability that the model made an error on a given sample.
    """

    @abstractmethod
    def predict(self, *, exog: pd.DataFrame) -> NDArray[np.float64]:
        """Predict calibrated uncertainty estimates.

        Args:
            exog: DataFrame containing raw uncertainty estimates or features
                (e.g., data uncertainty, knowledge uncertainty).

        Returns:
            NDArray: Calibrated uncertainty estimates as a 1D array of float values
                in the range [0, 1], where higher values indicate higher probability
                of error.
        """


class ProbabilityCalibrator(Calibrator, Protocol):
    """Protocol for calibrators that calibrate class probabilities.

    Probability calibrators transform raw classification model outputs
    (uncalibrated probabilities) into better-calibrated probability estimates
    where predicted probabilities more accurately reflect true likelihood of class membership.
    """

    @abstractmethod
    def predict_proba(self, *, exog: pd.DataFrame) -> NDArray[np.float64]:
        """Predict calibrated class probabilities.

        Args:
            exog: DataFrame containing uncalibrated class probability predictions from
                a classifier. For binary classification, should contain both class
                probabilities "proba_0" and "proba_1".

        Returns:
            NDArray: Calibrated class probabilities as a 2D array of shape
                (n_samples, n_classes) where each row contains non-negative values
                summing to 1.0.
        """
