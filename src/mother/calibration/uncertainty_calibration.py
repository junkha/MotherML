from typing import Self

import pandas as pd
import statsmodels as sm
from numpy.typing import NDArray
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from statsmodels.genmod.generalized_linear_model import GLM

from mother.calibration.calibration_interface import UncertaintyCalibrator


# BaggingRegressor + IR: ensemble of IR estimators with bootstrap sampling
# can estimate out-of-bag (OOB) generalization error
class BaggingIRErrorCalibrator(UncertaintyCalibrator):
    """Bagging ensemble of isotonic regressors for uncertainty calibration.

    Combines an ensemble of isotonic regression models trained on bootstrap samples
    to provide robust uncertainty calibration. Isotonic regression enforces monotonicity
    (increasing relationship) between input uncertainty estimates and true error rates.

    The bagging approach provides out-of-bag (OOB) error estimates automatically,
    which can be used to assess calibration quality.

    Attributes:
        n_estimators: Number of isotonic regression estimators in the ensemble.
        uncertainty_type: Name of the column in exog containing the uncertainty
            estimates to calibrate (e.g., "knowledge_uncertainty").
        oob_score: Out-of-bag score from the ensemble (set after calling fit).
    """

    def __init__(self, n_estimators: int = 100, uncertainty_type: str = "knowledge_uncertainty"):
        """Initialize the Bagging Isotonic Regression calibrator.

        Args:
            n_estimators: Number of bootstrap samples and isotonic regressors.
                Default is 100.
            uncertainty_type: Column name in the input DataFrame containing uncertainty
                estimate to calibrate. Default is "knowledge_uncertainty".
        """
        self.n_estimators = n_estimators
        self.uncertainty_type = uncertainty_type

    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the bagging ensemble of isotonic regressors.

        Args:
            exog: DataFrame containing uncertainty estimates with a column matching
                self.uncertainty_type.
            endog: Binary error indicators (0 = correct, 1 = error) corresponding
                to the samples in exog.

        Returns:
            self: The fitted calibrator object.
        """
        exog = exog[[self.uncertainty_type]]
        self.model_ = BaggingRegressor(
            estimator=IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip"),
            n_estimators=self.n_estimators,
            bootstrap=True,
            oob_score=True,
            random_state=42,
        )
        self.model_.fit(exog, endog)
        self.oob_score = self.model_.oob_score_
        return self

    def predict(self, *, exog: pd.DataFrame) -> NDArray:
        """Predict calibrated uncertainty estimates.

        Args:
            exog: DataFrame containing uncertainty estimates with a column matching
                self.uncertainty_type.

        Returns:
            NDArray: Calibrated uncertainty estimates as a 1D array with values in [0, 1],
            representing error likelihood per sample.
        """
        return self.model_.predict(exog[[self.uncertainty_type]])


# Random forest regressor with monotonic constraints and bootstrap sampling
# can estimate out-of-bag (OOB) generalization error
class RandomForestErrorCalibrator(UncertaintyCalibrator):
    """Random forest regressor with monotonic constraints for uncertainty calibration.

    Uses a random forest ensemble with monotonic increasing constraints to calibrate
    uncertainty estimates. All features are constrained to have increasing relationships
    with error probability.

    The forest-based approach automatically provides out-of-bag (OOB) error estimates
    for calibration quality assessment.

    Attributes:
        n_estimators: Number of trees in the random forest.
        oob_score: Out-of-bag score from the forest (set after calling fit)
    """

    def __init__(self, n_estimators: int = 1000):
        """Initialize the Random Forest error calibrator.

        Args:
            n_estimators: Number of trees in the forest. Default is 1000.
        """
        self.n_estimators = n_estimators

    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the random forest with monotonic constraints for all features.

        Args:
            exog: DataFrame containing multivariate uncertainty features.
            endog: Binary error indicators (0 = correct, 1 = error) corresponding
                to the samples in exog.

        Returns:
            self: The fitted calibrator object.
        """
        monotonic_constraints = [1] * exog.shape[1]  # all features increasing
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=42,
            bootstrap=True,
            oob_score=True,
            monotonic_cst=monotonic_constraints,
        ).fit(exog, endog)
        self.oob_score = self.model_.oob_score_
        return self

    def predict(self, *, exog: pd.DataFrame) -> NDArray:
        """Predict calibrated uncertainty estimates.

        Args:
            exog: DataFrame containing multivariate uncertainty features.

        Returns:
            NDArray: Calibrated uncertainty estimates as a 1D array with values in [0, 1].
        """
        return self.model_.predict(exog)


# GLM: multivariate GLM with both data and knowledge uncertainty as features
# exog: features for GLM (e.g. uncertainty estimates of type data and knowledge)
class GLMErrorCalibrator(UncertaintyCalibrator):
    """Generalized linear model (GLM) calibrator for uncertainty estimation.

    Fits a GLM with binomial family and logit link to calibrate uncertainty estimates.
    This multivariate approach can use multiple uncertainty features simultaneously
    (e.g., knowledge uncertainty, data uncertainty) to predict error probability.
    """

    def __init__(self):
        """Initialize the GLM error calibrator with binomial family."""
        self.family = sm.genmod.families.Binomial(link=sm.genmod.families.links.Logit())

    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the GLM calibration model.

        Adds an intercept term to the features and fits a binomial GLM with logit link.

        Args:
            exog: DataFrame containing multivariate uncertainty features
                (e.g., knowledge_uncertainty, data_uncertainty).
            endog: Binary error indicators (0 = correct, 1 = error) corresponding
                to the samples in exog.

        Returns:
            self: The fitted calibrator object.
        """
        exog = sm.tools.add_constant(exog, has_constant="add")
        self.model_ = GLM(endog, exog, family=self.family, missing="raise").fit()
        return self

    def predict(self, *, exog: pd.DataFrame) -> NDArray:
        """Predict calibrated uncertainty estimates using the GLM.

        Args:
            exog: DataFrame containing multivariate uncertainty features.

        Returns:
            NDArray: Predicted error probabilities (calibrated uncertainty) as a 1D
                array with values in [0, 1].
        """
        exog = sm.tools.add_constant(exog, has_constant="add")
        return self.model_.predict(exog)


class IsotonicRegressionErrorCalibrator(UncertaintyCalibrator):
    """Univariate isotonic regression calibrator for uncertainty estimation.

    Fits an isotonic (monotonically increasing) regression model to calibrate a single
    uncertainty feature.

    Attributes:
        uncertainty_type: Name of the column to use as the uncertainty feature
            (default: "knowledge_uncertainty").
    """

    def __init__(self, uncertainty_type: str = "knowledge_uncertainty"):
        """Initialize the isotonic regression calibrator.

        Args:
            uncertainty_type: Column name containing the univariate uncertainty
                feature to calibrate. Default is "knowledge_uncertainty".
        """
        self.uncertainty_type = uncertainty_type

    def fit(self, *, exog: pd.DataFrame, endog: NDArray | pd.Series) -> Self:
        """Fit the isotonic regression model.

        Args:
            exog: DataFrame containing uncertainty estimates with a column matching
                self.uncertainty_type.
            endog: Binary error indicators (0 = correct, 1 = error) corresponding
                to the samples in exog.

        Returns:
            self: The fitted calibrator object.
        """
        exog_by_type = exog[self.uncertainty_type]
        self.model_ = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
        self.model_.fit(exog_by_type, endog)
        return self

    def predict(self, *, exog: pd.DataFrame) -> NDArray:
        """Predict calibrated uncertainty estimates using isotonic regression.

        Args:
            exog: DataFrame containing uncertainty estimates with a column matching
                self.uncertainty_type.

        Returns:
            NDArray: Calibrated uncertainty estimates as a 1D array clipped to [0, 1].
        """
        return self.model_.predict(exog[self.uncertainty_type])
