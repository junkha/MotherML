"""Calibration module for transforming model outputs into reliable estimates.

This module provides calibration techniques for both probability predictions and
uncertainty quantification. Calibration is the process of transforming raw model outputs
(uncalibrated probabilities or uncertainty estimates) into well-calibrated estimates
that accurately reflect the true reliability or correctness of predictions.

Submodules:
    calibration_interface: Abstract Protocol definitions for calibrators.
    probability_calibration: Calibration methods for classification probabilities
        (Platt scaling, Venn-Abers).
    uncertainty_calibration: Calibration methods for prediction uncertainty
        (Isotonic regression, Random Forest, GLM, Bagging ensembles).

Key Concepts:
    - **Probability Calibration**: Transforms classifier output probabilities to match
      actual class membership likelihood.
    - **Uncertainty Calibration**: Transforms uncertainty estimates to match true
      probability of model error.

Example:
    >>> from mother.calibration.probability_calibration import PlattProbabilityCalibrator
    >>> calibrator = PlattProbabilityCalibrator()
    >>> calibrator.fit(exog=uncalibrated_proba, endog=true_labels)
    >>> calibrated_proba = calibrator.predict_proba(exog=uncalibrated_proba)

    >>> from mother.calibration.uncertainty_calibration import IsotonicRegressionErrorCalibrator
    >>> unc_calibrator = IsotonicRegressionErrorCalibrator()
    >>> unc_calibrator.fit(exog=uncertainty_estimates, endog=error_indicators)
    >>> calibrated_uncertainty = unc_calibrator.predict(exog=uncertainty_estimates)
"""
