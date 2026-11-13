from typing import Self

import numpy as np
import pandas as pd
import statsmodels as sm
from numpy.typing import NDArray
from statsmodels.genmod.generalized_linear_model import GLM
from venn_abers import VennAbersCalibrator

from mother.calibration.calibration_interface import ProbabilityCalibrator


class PlattProbabilityCalibrator(ProbabilityCalibrator):
    """Platt scaling calibrator using logistic regression.

    Platt scaling fits a logistic regression model to map uncalibrated probabilities
    to calibrated probabilities. It uses only the probability of class 1 as the feature
    and learns a scaling transformation via generalized linear model (GLM) with
    binomial family and logit link function.

    This method is particularly effective when the uncalibrated probabilities are
    already well-separated but poorly calibrated (e.g., overconfident).
    """

    def __init__(self):
        link_func = sm.genmod.families.links.Logit()
        self.family = sm.genmod.families.Binomial(link=link_func)

    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the Platt scaling model.

        Args:
            exog: DataFrame containing uncalibrated class probabilities with a "proba_1"
                column representing the probability of the positive class.
            endog: True binary labels (0 or 1) corresponding to the samples in exog.

        Returns:
            self: The fitted calibrator object.

        Raises:
            KeyError: If "proba_1" column is not present in exog.
        """
        exog = exog[["proba_1"]]
        exog = sm.tools.add_constant(exog, has_constant="add")
        self.model_ = GLM(endog, exog, family=self.family, missing="raise").fit()
        return self

    def predict_proba(self, *, exog: pd.DataFrame) -> NDArray:
        """Predict calibrated probabilities using the fitted Platt scaler.

        Args:
            exog: DataFrame containing uncalibrated class probabilities with a "proba_1"
                column representing the probability of the positive class.

        Returns:
            NDArray: A 2D array of shape (n_samples, 2) where:
                - Column 0: Calibrated probability of class 0
                - Column 1: Calibrated probability of class 1
        """
        exog = exog[["proba_1"]]
        exog = sm.tools.add_constant(exog, has_constant="add")
        proba_1_scaled = self.model_.predict(exog)
        proba_0_scaled = 1 - proba_1_scaled
        probas_scaled = np.column_stack((proba_0_scaled, proba_1_scaled))
        return probas_scaled


# Venn-Abers Predictor: combines two isotonic regression models (one for each class)
# exog: binary class probabilities
class VennAbersProbabilityCalibrator(ProbabilityCalibrator):
    """Venn-Abers predictor for probability calibration.

    The Venn-Abers predictor is a non-conformity-based calibration method that combines
    two isotonic regression models (one for each class) to provide well-calibrated
    probability estimates. It's based on conformal prediction principles and offers
    theoretical guarantees on calibration.
    """

    def __init__(self):
        """Initialize the Venn-Abers calibrator."""

    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the Venn-Abers calibration model. Stores the calibration set data (uncalibrated
        probabilities and true labels) for use during prediction. The Venn-Abers method uses
        this data to compute conformal predictions.

        Args:
            exog: DataFrame containing uncalibrated binary class probabilities.
                Should have shape (n_samples, 2).
            endog: True binary labels (0 or 1) corresponding to the samples in exog.

        Returns:
            self: The fitted calibrator object.
        """
        self.model_ = VennAbersCalibrator()
        self.exog_ = exog.to_numpy()
        self.endog_ = endog.values if isinstance(endog, pd.Series) else np.asarray(endog)
        return self

    def predict_proba(self, *, exog: pd.DataFrame) -> NDArray:
        """Predict calibrated probabilities using Venn-Abers method.

        Args:
            exog: DataFrame containing uncalibrated binary class probabilities for
                which calibrated predictions are desired.

        Returns:
            NDArray: A 2D array of calibrated class probabilities for each test sample.
                Shape is (n_test_samples, 2) with columns for calibrated probabilities of
                class 0 and class 1 respectively.
        """
        exog_array = exog.to_numpy()
        return self.model_.predict_proba(p_cal=self.exog_, y_cal=self.endog_, p_test=exog_array)
