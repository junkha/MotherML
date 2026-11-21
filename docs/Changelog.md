# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- version list -->

## v3.7.3 (2026-11-21)

### Chores

- **deps**: Bump tabpfn from 8.2.0 to 10.1.0
  ([#159](https://github.com/Bayer-Group/MotherML/pull/159),
  [`4ddacf0`](https://github.com/Bayer-Group/MotherML/commit/4ddacf0816ab8047c1fb38b3ea7f7b52d957c42f))


## v3.7.2 (2026-11-20)

### Bug Fixes

- **cv**: Set class attribute self.name for TimeSeriesGrouping
  ([#162](https://github.com/Bayer-Group/MotherML/pull/162),
  [`2fd9b02`](https://github.com/Bayer-Group/MotherML/commit/2fd9b024fcd741bcd0ea100ed2ce480f8e2f71cd))


## v3.7.1 (2026-11-20)

### Chores

- **deps**: Bump plotly from 6.9.0 to 7.1.0
  ([#161](https://github.com/Bayer-Group/MotherML/pull/161),
  [`bf712bd`](https://github.com/Bayer-Group/MotherML/commit/bf712bd7554839ff3b285066e095453759fe7fd2))


## v3.7.0 (2026-11-20)

### Documentation

- **ml**: Explain Greedy vs standard Boruta and test multi-target
  ([#155](https://github.com/Bayer-Group/MotherML/pull/155),
  [`ad42961`](https://github.com/Bayer-Group/MotherML/commit/ad429617b83ab7a748233518a11525bf19fc03a0))

### Features

- **ml**: Add Greedy Boruta as default Boruta variant
  ([#155](https://github.com/Bayer-Group/MotherML/pull/155),
  [`ad42961`](https://github.com/Bayer-Group/MotherML/commit/ad429617b83ab7a748233518a11525bf19fc03a0))

- **ml**: Add Greedy Boruta as default Boruta variant (#77)
  ([#155](https://github.com/Bayer-Group/MotherML/pull/155),
  [`ad42961`](https://github.com/Bayer-Group/MotherML/commit/ad429617b83ab7a748233518a11525bf19fc03a0))

### Testing

- **ml**: Cover Boruta variants on multi-target data
  ([#155](https://github.com/Bayer-Group/MotherML/pull/155),
  [`ad42961`](https://github.com/Bayer-Group/MotherML/commit/ad429617b83ab7a748233518a11525bf19fc03a0))


## v3.6.0 (2026-11-19)

### Bug Fixes

- **node**: Align NODE classifier predict_uncertainty with deterministc predict_proba
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **node**: Preserve multi-label column contract for backward compatibility
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **node**: Report multi-label probabilities separately
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

### Code Style

- **test**: Remove blank line ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

### Documentation

- **examples**: Adapt tutorial documentation following consecutive numbering of notebooks
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Add class relation overview for ensemble class families
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Add example notebook for usage of CVEnsembleClassifierMother
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Demonstrate ensemble subclass differences
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Demonstrate ensemble usage within Mother pipeline
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Improve documentation of ensemble training and evaluation
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Improve readability ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **examples**: Update tutorial to use renamed Lasso classifier class
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **readme**: Define fitted estimator variable returned from mother_cv before using it in ensemble
  documentation ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **readme**: Update cross-validation section with CV ensemble example
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

### Features

- **ml**: Accept member estimators without feature-input attributes
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Accept string type uncertainty methods in derived ensembles
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Adapt predict_uncertainty scheme to increase Mother compatibility of cv ensemble
  classifiers ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Choose available uncertainty type to return for opt
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Derive feature_names_in_ and n_features_in_ from member estimators
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Ensemble classifiers based on MotherCV and examples
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Ensure homogeneous ensembles ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Ensure non-negative uncertainty scores from ensemble-derived estimation via entropy
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Ensure scikit-learn compatibility
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Estimate knowledge uncertainty from averaged individual member uncertainties with
  'member_average' method ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Expose fitted attributes immediately after successful construction
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Extract member proba columns by name
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Fix compatibility with parent classes
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Fix default values and enhance zero-safe uncertainty prediction
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Improve member probas shape validation
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Introduce ensemble classifiers by combining fitted estimators from mother_cv
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Rename subclasses for clear usability
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Restructure uncertainty prediction
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Return unedited instance on fit only when no data is passed
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Split ensemble class into member-averaging and ensemble-derived subclass
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **ml**: Use string type uncertainty method in predict_uncertainty
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

### Testing

- **unit**: Add tests for error handling ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **unit**: Cover model persistance and loading with .pkl file
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))

- **unit**: Update tests for subclass functionality
  ([#152](https://github.com/Bayer-Group/MotherML/pull/152),
  [`5b3986a`](https://github.com/Bayer-Group/MotherML/commit/5b3986a1a84a5941e1e02dab9d29f3b1639cacba))


## v3.5.0 (2026-11-17)

### Bug Fixes

- **cv**: Align plot_cv_indices group lines with group edges
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **cv**: Lazy-import umap in report helpers to avoid TF/torch hang
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **cv**: Skip group boundary lines in plot_cv_indices for many groups
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **models**: Clear stale ARD state and harden Lasso target_type
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

### Documentation

- **examples**: Fix multitask random forest notebook
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **examples**: Re-run Lasso classification notebook
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **examples**: Run Boruta notebook and fix selection comparison
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

### Features

- **models**: Keep Lasso target type and backend parameters across get, set and clone
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **models**: Merge Lasso classifiers and add multi-target and ARD modes
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))

- **models**: Merge Lasso classifiers and add multi-target and ARD modes (#124)
  ([#151](https://github.com/Bayer-Group/MotherML/pull/151),
  [`4f5ae56`](https://github.com/Bayer-Group/MotherML/commit/4f5ae560f390663ac1ada53b9b5839d58fe63a1c))


## v3.4.0 (2026-11-17)

### Bug Fixes

- **cv**: Draw plot_cv_indices as contiguous bars sorted by group (#50, #51)
  ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))

- **cv**: Size group labels from consumed molecules, not exhausted iterators
  ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))

- **models**: Leave TabICL n_estimators fixed during Optuna tuning
  ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))

### Documentation

- **examples**: Refresh RF and settings notebooks for plot_cv_indices (#50, #51)
  ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))

### Features

- **cv**: HDBSCAN outlier singletons, clearer plot_cv_indices, TabICL Optuna space, and NODE CV
  tests ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))

- **cv**: Optional HDBSCAN outlier singletons for GroupKFold
  ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))

### Testing

- **models**: Cover NODE mother_cv, MotherTuner, and nested CV paths
  ([#153](https://github.com/Bayer-Group/MotherML/pull/153),
  [`6b05096`](https://github.com/Bayer-Group/MotherML/commit/6b050960da858d794e7e98a403a450df47246ba2))


## v3.3.3 (2026-11-16)

### Documentation

- **examples**: Combine the uncertainty tutorials into one notebook
  ([#149](https://github.com/Bayer-Group/MotherML/pull/149),
  [`5a67f6c`](https://github.com/Bayer-Group/MotherML/commit/5a67f6c90a9d64b613f7ed305701b6dcb03e912e))


## v3.3.2 (2026-11-16)

### Chores

- **deps**: Bump pypa/gh-action-pypi-publish from 1.14.2 to 1.14.3
  ([#139](https://github.com/Bayer-Group/MotherML/pull/139),
  [`4e065a8`](https://github.com/Bayer-Group/MotherML/commit/4e065a89b62682d7b28aa85875f83f7f229ec6c0))


## v3.3.1 (2026-11-16)

### Chores

- **deps**: Bump torch from 2.14.1 to 2.15.0
  ([#141](https://github.com/Bayer-Group/MotherML/pull/141),
  [`5aa06fb`](https://github.com/Bayer-Group/MotherML/commit/5aa06fb526980db77edd8d02d87110327a8f9c61))


## v3.3.0 (2026-11-15)

### Bug Fixes

- **catboost**: Apply prediction_type per GP posterior sample
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

### Documentation

- **examples**: Add Gaussian-process CatBoost uncertainty notebook
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **examples**: Expand explanation and evaluation in GP CatBoost notebook
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **examples**: Replace exact GP comparison with usage guidance in GP CatBoost notebook
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **examples**: Use merged posterior samples in GP CatBoost notebook
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **GPcatboost**: Improve the gaussian catboost notebook with better examples
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **notebook**: Compare CatBoost uncertainties with Spearman, Kendall and Pearson
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **notebooks**: Fix the quantile and GP sections of the uncertainty notebook
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **notebooks**: Update GP CatBoost notebook to configurable iterations
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

### Features

- **catboost**: Merge GP CatBoost posterior samples into one model via sum_models
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **models**: GP CatBoost as a single CatBoostRegressor-compatible model (#125)
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **models**: Make GP CatBoost behave like CatBoostRegressor
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

### Refactoring

- **catboost**: Complete GP type hints and move NODE BALSA-EMD section
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

- **catboost**: Drop legacy GP pickle support and document threads
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))

### Testing

- **catboost**: Check GP tree ranges through pickle, copy and clone
  ([#136](https://github.com/Bayer-Group/MotherML/pull/136),
  [`6d54e63`](https://github.com/Bayer-Group/MotherML/commit/6d54e6330d99de5ac8c50829b2e724f23a5d6f43))


## v3.2.2 (2026-11-15)

### Chores

- **deps-dev**: Bump mypy from 2.4.0 to 2.4.1
  ([#138](https://github.com/Bayer-Group/MotherML/pull/138),
  [`8d672c2`](https://github.com/Bayer-Group/MotherML/commit/8d672c2d1c60e1c38037b820bed9acfd7452e40c))


## v3.2.1 (2026-11-15)

### Chores

- **deps-dev**: Bump poethepoet from 0.48.0 to 0.49.0
  ([#143](https://github.com/Bayer-Group/MotherML/pull/143),
  [`464a0c6`](https://github.com/Bayer-Group/MotherML/commit/464a0c6564469d99a9a44a0c175de8845620d1a5))


## v3.2.0 (2026-11-14)

### Bug Fixes

- **catboost**: Expose bootstrap tuning on ranker
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

- **catboost**: Expose classifier bootstrap tuning flag
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

- **catboost**: Scope bootstrap-level tuning to estimators that use it
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

- **tabpfn**: Disable autocast for pre-fitted embedding extraction
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

### Code Style

- **catboost**: Reformat bootstrap tuning signatures
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

### Features

- **catboost**: Introduce optional bootstrap level tuning
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

### Testing

- **catboost**: Cover classifier bootstrap level tuning
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))

- **catboost**: Verify bootstrap tuning search space distributions
  ([#65](https://github.com/Bayer-Group/MotherML/pull/65),
  [`55c5703`](https://github.com/Bayer-Group/MotherML/commit/55c57030a7904a9b7799e00061662fad12797b37))


## v3.1.3 (2026-11-14)

### Chores

- **deps**: Bump tornado from 6.6 to 6.6.1
  ([#145](https://github.com/Bayer-Group/MotherML/pull/145),
  [`534c3e9`](https://github.com/Bayer-Group/MotherML/commit/534c3e91350c48580131072a3c2b788bd882a962))


## v3.1.2 (2026-11-13)

### Continuous Integration

- **tests**: Restore parallel slow tests and fix xdist worker count (#137)
  ([#144](https://github.com/Bayer-Group/MotherML/pull/144),
  [`a30a2bb`](https://github.com/Bayer-Group/MotherML/commit/a30a2bbe59bab2962fcbcd97ca6b41b8a716351e))

- **tests**: Run non-NODE slow tests in parallel again, keep NODE serial
  ([#144](https://github.com/Bayer-Group/MotherML/pull/144),
  [`a30a2bb`](https://github.com/Bayer-Group/MotherML/commit/a30a2bbe59bab2962fcbcd97ca6b41b8a716351e))

- **tests**: Use logical CPUs for pytest-xdist workers
  ([#144](https://github.com/Bayer-Group/MotherML/pull/144),
  [`a30a2bb`](https://github.com/Bayer-Group/MotherML/commit/a30a2bbe59bab2962fcbcd97ca6b41b8a716351e))

### Testing

- **ranking**: Shrink feature-selection models in slow ranking tests
  ([#144](https://github.com/Bayer-Group/MotherML/pull/144),
  [`a30a2bb`](https://github.com/Bayer-Group/MotherML/commit/a30a2bbe59bab2962fcbcd97ca6b41b8a716351e))


## v3.1.1 (2026-11-13)

### Bug Fixes

- **optimization**: Fail loudly on early stopping misconfiguration
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

- **optimization**: Guard terminator reporting on usable fold scores
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

- **optimization**: Raise when early stopping is requested without torch
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

- **optimization**: Support hold-out CV in Optuna tuning
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

- **optimization**: Support hold-out CV in Optuna tuning path
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

- **tabpfn**: Disable autocast for pre-fitted embeddings
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

### Documentation

- **optimization**: Clarify Optuna callback disablement cases
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))

### Testing

- **tabpfn**: Remove internal autocast assertion
  ([#58](https://github.com/Bayer-Group/MotherML/pull/58),
  [`30a3fc0`](https://github.com/Bayer-Group/MotherML/commit/30a3fc09cc2efa6d22df43867c686db2abeef01b))


## v3.1.0 (2026-11-12)

### Bug Fixes

- **config**: Default native feature declarations to empty lists
  ([#114](https://github.com/Bayer-Group/MotherML/pull/114),
  [`7bec971`](https://github.com/Bayer-Group/MotherML/commit/7bec971458b12b908e5fe86cb1bfcac7589bb521))

- **pipeline**: Preserve integer categorical features during selection
  ([#114](https://github.com/Bayer-Group/MotherML/pull/114),
  [`7bec971`](https://github.com/Bayer-Group/MotherML/commit/7bec971458b12b908e5fe86cb1bfcac7589bb521))

### Features

- **catboost**: Support text and embedding feature configuration
  ([#114](https://github.com/Bayer-Group/MotherML/pull/114),
  [`7bec971`](https://github.com/Bayer-Group/MotherML/commit/7bec971458b12b908e5fe86cb1bfcac7589bb521))


## v3.0.3 (2026-11-12)

### Testing

- **node**: Reduce uncertainty sample counts in slow NODE tests
  ([#132](https://github.com/Bayer-Group/MotherML/pull/132),
  [`24d0b0f`](https://github.com/Bayer-Group/MotherML/commit/24d0b0f51238210e2151615d890e3f8a0493b1b0))


## v3.0.2 (2026-11-12)

### Documentation

- **readme**: Update outdated README with current models, extras and uncertainty (#133)
  ([#134](https://github.com/Bayer-Group/MotherML/pull/134),
  [`763cb16`](https://github.com/Bayer-Group/MotherML/commit/763cb16893ac8e1891d8a2c34ef5d4c4f84aa0c0))


## v3.0.1 (2026-11-12)

### Chores

- **deps**: Bump aiohttp from 3.14.3 to 3.14.4
  ([#127](https://github.com/Bayer-Group/MotherML/pull/127),
  [`91a764b`](https://github.com/Bayer-Group/MotherML/commit/91a764bab558fb1df652aadf9ca9ef0aa392ea6d))


## v3.0.0 (2026-11-11)

### Bug Fixes

- **chemeleon**: Validate parameters when they are used, not in __init__
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **ci**: Grant pages read permission to docs build job
  ([`2b027f4`](https://github.com/Bayer-Group/MotherML/commit/2b027f44b08c06cfb59164466449486e03eb40bd))

- **deps**: Cap rdkit below 2026.9 for the chemprop extra
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **ml**: Correct NODE and CheMeleon workflows
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Address PR review feedback ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Align uncertainty output with Mother standard and follow sample dtype
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Drop TorchScript from the Entmoid15 activation
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Follow Mother multi-target uncertainty schema and ignore float64 default
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Handle None hidden dimensions in MLP heads
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Harden NODE inputs, params, tuning and uncertainty
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Invalidate initialization cache when loading model state
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Keep encoders on warm start, reject cat_features for arrays, drop dead NODEModel
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Keep the shared NODE base out of model discovery
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Make BALSA-EMD scores batch-independent and accept tuple MLP dims
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Move all shared tensor references to the target device
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Preserve distribution shapes during device traversal
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Preserve ODST output dimensions for higher-rank inputs
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Raise the standard extras error for zuko and speed up init
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Reject boolean dropout rates ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Remove redundant RealNVP flow option
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Repair notebook rendering and refresh tutorial indexes
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Tolerate single-row training batches with batch normalization
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Validate batch_size and follow the default dtype in ODST
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Validate dropout settings and guard uncertainty inference
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Validate inference rows instead of sampler types
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node,chemeleon**: Address PR #71 review comments
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node,chemeleon**: Address review findings in NODE and CheMeleon
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **notebook**: Format CheMeleon install cell
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **tabpfn**: Disable autocast for pre-fitted embeddings
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Build System

- **deps**: Cap rdkit below 2026.9 for every install
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **deps**: Require chemprop>=2.2, add real CheMeleon test
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Chores

- **ml**: Defer registry prefix fix to PR #91
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Continuous Integration

- Run the CheMeleon test in the slow test job
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Documentation

- **examples**: Demonstrate CheMeleon embeddings with CatBoost
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **examples**: List the CheMeleon quick-start notebook in both indexes
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **models**: Document NODE and TabICL model backends
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Clarify BALSA-EMD distributional disagreement
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Fix NODE notebook errors and add missing docstrings
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Features

- **chemeleon**: Support configurable embedding output layouts
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Add CheMeleon fingerprints and NODE regression workflows
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Add configurable tree chunk size
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Standardize flow targets and preserve output units
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Standardize regression targets for non-flow heads
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Performance Improvements

- **node**: Chunk feature selection and data-aware initialization
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

### Testing

- Add a slow integration test that saves a small real
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- Exclude only NODE from the generic algorithm sweeps and fix the reason
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **ml**: Restore NODE coverage in registry-lookup test
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **node**: Silence expected skorch shuffle and requires_grad warnings
  ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))

- **tabpfn**: Align tests with main ([#71](https://github.com/Bayer-Group/MotherML/pull/71),
  [`ff8ecf2`](https://github.com/Bayer-Group/MotherML/commit/ff8ecf291e6c09be0b498591f23ac18f529616b0))


## v2.1.11 (2026-11-08)

### Chores

- **deps-dev**: Bump ruff from 0.16.10 to 0.17.3
  ([#107](https://github.com/Bayer-Group/MotherML/pull/107),
  [`af68bb6`](https://github.com/Bayer-Group/MotherML/commit/af68bb687707926769cc29ffc7252096ebb2b833))


## v2.1.10 (2026-11-07)

### Chores

- **deps**: Bump rdkit from 2026.3.6 to 2026.9.1
  ([#108](https://github.com/Bayer-Group/MotherML/pull/108),
  [`c3499a8`](https://github.com/Bayer-Group/MotherML/commit/c3499a84cfca0de409e735a8282a6cf120bbeaf2))


## v2.1.9 (2026-11-07)

### Chores

- **deps-dev**: Update uv-build requirement
  ([#109](https://github.com/Bayer-Group/MotherML/pull/109),
  [`65d2d98`](https://github.com/Bayer-Group/MotherML/commit/65d2d9803c984c5bb95880fc7692101738a10a3f))


## v2.1.8 (2026-11-07)

### Chores

- **deps**: Bump actions/upload-artifact from 7.0.1 to 7.0.2
  ([#110](https://github.com/Bayer-Group/MotherML/pull/110),
  [`56585ee`](https://github.com/Bayer-Group/MotherML/commit/56585eeff9560d5a66c3a3f54d07c6242dec88d9))


## v2.1.7 (2026-11-06)

### Chores

- **deps**: Bump actions/download-artifact from 8.0.1 to 8.0.2
  ([#111](https://github.com/Bayer-Group/MotherML/pull/111),
  [`ce9698e`](https://github.com/Bayer-Group/MotherML/commit/ce9698e2a42d93cc36614eba1fc02f7654e32d0c))


## v2.1.6 (2026-11-06)

### Chores

- **deps**: Bump numpy from 2.2.6 to 2.4.6
  ([#112](https://github.com/Bayer-Group/MotherML/pull/112),
  [`9ef0fe1`](https://github.com/Bayer-Group/MotherML/commit/9ef0fe11482a3b9b579e1f43c942ebe0491e8b66))


## v2.1.5 (2026-11-06)

### Chores

- **deps**: Bump gitpython from 3.2.0 to 3.2.1
  ([#123](https://github.com/Bayer-Group/MotherML/pull/123),
  [`665bcc5`](https://github.com/Bayer-Group/MotherML/commit/665bcc5e8d709dc4445de41f362f0b852f48b248))


## v2.1.4 (2026-11-06)

### Chores

- **deps**: Bump filelock from 4.0.11 to 4.2.1
  ([#117](https://github.com/Bayer-Group/MotherML/pull/117),
  [`0c7459f`](https://github.com/Bayer-Group/MotherML/commit/0c7459fa5f566b3c3876af8437eed3964dbcaf20))


## v2.1.3 (2026-11-06)

### Chores

- **deps**: Bump pillow from 12.3.0 to 13.0.0
  ([#116](https://github.com/Bayer-Group/MotherML/pull/116),
  [`ac2dede`](https://github.com/Bayer-Group/MotherML/commit/ac2dedef4ecd529eb5d01ff7226ce1843fdf5e14))


## v2.1.2 (2026-11-05)

### Chores

- **deps**: Bump certifi from 2026.7.22 to 2026.10.18
  ([#120](https://github.com/Bayer-Group/MotherML/pull/120),
  [`551c4da`](https://github.com/Bayer-Group/MotherML/commit/551c4daaa1f496a52d49dc114b2742424916c9de))

- **deps**: Bump virtualenv from 21.14.5 to 21.14.13
  ([#118](https://github.com/Bayer-Group/MotherML/pull/118),
  [`ed39863`](https://github.com/Bayer-Group/MotherML/commit/ed39863ea1cfcb16de10129de90eee566f8f0972))


## v2.1.1 (2026-11-05)

### Chores

- **deps**: Bump tornado from 6.5.10 to 6.6
  ([#121](https://github.com/Bayer-Group/MotherML/pull/121),
  [`0014443`](https://github.com/Bayer-Group/MotherML/commit/001444312d57f2c35a1bd38333e475b8a20228b9))


## v2.1.0 (2026-10-28)

### Bug Fixes

- Derive model registry algorithm names via prefix removal
  ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))

- **cv**: Make MurckoScaffoldGroupingFromMols fit check effective
  ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))

### Code Style

- Apply isort and ruff formatting ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))

- **cv**: Fix import formatting in cv/__init__.py
  ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))

### Features

- **cv**: Add Murcko scaffold grouping transformer
  ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))

### Testing

- **cv**: Add smoke test for Murcko scaffold grouping
  ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))

- **cv**: Cover Murcko scaffold grouping behavior
  ([#106](https://github.com/Bayer-Group/MotherML/pull/106),
  [`d339a04`](https://github.com/Bayer-Group/MotherML/commit/d339a04da483045c8a9a76eb136d44c7a4159b6c))


## v2.0.5 (2026-10-16)

### Documentation

- Add TabICLEmbeddingTransformer to the file
  ([#94](https://github.com/Bayer-Group/MotherML/pull/94),
  [`1159b10`](https://github.com/Bayer-Group/MotherML/commit/1159b10200c035a06857f0203f807da13d0a1d5b))

- **tabicl**: Add missing documentation for TabICL
  ([#94](https://github.com/Bayer-Group/MotherML/pull/94),
  [`1159b10`](https://github.com/Bayer-Group/MotherML/commit/1159b10200c035a06857f0203f807da13d0a1d5b))

- **tabicl**: Fix capitalization and formatting for TabICL entry
  ([#94](https://github.com/Bayer-Group/MotherML/pull/94),
  [`1159b10`](https://github.com/Bayer-Group/MotherML/commit/1159b10200c035a06857f0203f807da13d0a1d5b))


## v2.0.4 (2026-10-13)

### Bug Fixes

- Derive model registry algorithm names via prefix removal
  ([#91](https://github.com/Bayer-Group/MotherML/pull/91),
  [`dc9c1d8`](https://github.com/Bayer-Group/MotherML/commit/dc9c1d8258afd86ceaa54c6e91c786200321a1f5))

### Code Style

- Apply isort and ruff formatting ([#91](https://github.com/Bayer-Group/MotherML/pull/91),
  [`dc9c1d8`](https://github.com/Bayer-Group/MotherML/commit/dc9c1d8258afd86ceaa54c6e91c786200321a1f5))


## v2.0.3 (2026-10-12)

### Bug Fixes

- **ml**: Bugfix in mother_cv when tuning is disabled
  ([#102](https://github.com/Bayer-Group/MotherML/pull/102),
  [`e3a7ced`](https://github.com/Bayer-Group/MotherML/commit/e3a7ced74cd7c2e47010783d687849232c9d4200))

- **ml**: Return distinct estimators with mother_cv if tuning is disabled
  ([#102](https://github.com/Bayer-Group/MotherML/pull/102),
  [`e3a7ced`](https://github.com/Bayer-Group/MotherML/commit/e3a7ced74cd7c2e47010783d687849232c9d4200))

- **test**: Adapt unit test to changes in mother_cv
  ([#102](https://github.com/Bayer-Group/MotherML/pull/102),
  [`e3a7ced`](https://github.com/Bayer-Group/MotherML/commit/e3a7ced74cd7c2e47010783d687849232c9d4200))


## v2.0.2 (2026-10-05)

### Bug Fixes

- Changelog creation and missing changelog added
  ([#95](https://github.com/Bayer-Group/MotherML/pull/95),
  [`be52939`](https://github.com/Bayer-Group/MotherML/commit/be52939c5a9c4de0999cb94ca83dfb0998ce9046))

- No manual version bump ([#95](https://github.com/Bayer-Group/MotherML/pull/95),
  [`be52939`](https://github.com/Bayer-Group/MotherML/commit/be52939c5a9c4de0999cb94ca83dfb0998ce9046))

- Update build command to regenerate and stage uv.lock for release commits
  ([#95](https://github.com/Bayer-Group/MotherML/pull/95),
  [`be52939`](https://github.com/Bayer-Group/MotherML/commit/be52939c5a9c4de0999cb94ca83dfb0998ce9046))

- Update build command to regenerate and stage uv.lock for release…
  ([#95](https://github.com/Bayer-Group/MotherML/pull/95),
  [`be52939`](https://github.com/Bayer-Group/MotherML/commit/be52939c5a9c4de0999cb94ca83dfb0998ce9046))

### Chores

- Uv audit and fix project version ([#95](https://github.com/Bayer-Group/MotherML/pull/95),
  [`be52939`](https://github.com/Bayer-Group/MotherML/commit/be52939c5a9c4de0999cb94ca83dfb0998ce9046))


## v2.0.1 (2026-10-01)

### Bug Fixes

- Reintegrate release into workflow.yml
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

- Update job dependencies in workflow.yml for correct execution order
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

### Chores

- Fix CI/CD workflows for release and manual PyPI publishing
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

- Update action versions and set timeouts for workflows
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

### Continuous Integration

- Enhance workflow validation and release process
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

---

**Detailed Changes**: [v2.0.0...v2.0.1](https://github.com/Bayer-Group/MotherML/compare/v2.0.0...v2.0.1)


## v2.0.0 (2026-09-29)

### Bug Fixes

- **catboost**: Fix loss_function=None handling, set_params rollback, and doc example
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost**: Harden ranker loss-function handling and GP regressor tuning
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost**: Reject top=/mode=Classic conflicts, fix stale top/max_pairs, fix avg_ndcg_score direction
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Apply mode=NDCG fix-up to top= embedded via set_params
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Compare embedded loss-string params numerically
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Make set_params atomic and validate ranking array shapes
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Require explicit mode for embedded top=, revert avg_ndcg_score shift
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Validate malformed top/max_pairs tokens, fix avg_ndcg_score DataFrame iteration
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Validate top/max_pairs family gaps, dedupe group-indexing
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking,tabpfn**: Harden loss-function validation, atomic set_params, uncertainty wrapper consistency
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ml**: Stop GP unpickling from re-enabling disabled tuning flags
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Enable metadata routing in tutorial, document pred normalization
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Reject 2-D group_id/group_ids instead of silently flattening
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Reject missing values in avg_ndcg_score groups
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Reject zero-column ensemble arrays, document larger-is-better convention
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Validate k strictly and clean up top-k analysis docs/notebook
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Validate missing group IDs in top-k analysis
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **tabpfn**: Honor configured device for pre-fitted models
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **tabpfn**: Validate pre-fitted model before embedding extraction
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Chores

- Fix example notebook formatting and nbqa check
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Remove deprecated unit tests for example notebooks
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Remove unmaintained polaris-lib dependency and fix example notebooks (still issues)
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Speedup env creation for style check
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Update actions to latest versions in workflow
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Update workflow for Python version and remove test-pypi-publish job
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Uv audit
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Remove PR status doc, not meant for the repo
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Documentation

- Add CI/CD & Release Overview to documentation
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Document larger-is-better convention in avg_ndcg_score
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Update PR status doc for top=/mode fix, mother_cv note, test fixes
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **tabpfn**: Clarify device/precision are not enforced for pre-fitted models
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Features

- **ranking**: Add CatBoost ranker tuning and uncertainty workflows
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Add functions to convert scores to ranks and score matrices to ranks
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Refactoring

- Remove wrong _TransformOnlyValidMols usage
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost**: Move ranking utility functions to utils module
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

---

**Detailed Changes**: [v1.2.0...v2.0.0](https://github.com/Bayer-Group/MotherML/compare/v1.2.0...v2.0.0)

**Resolved Issues**: [#25](https://github.com/Bayer-Group/MotherML/issues/25),
[#33](https://github.com/Bayer-Group/MotherML/issues/33),
[#57](https://github.com/Bayer-Group/MotherML/issues/57).


## v1.2.0 (2026-09-01)

### Bug Fixes

- Potential fix for pull request finding
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Remove changelog file reference from semantic release configuration
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Update actions to latest versions in workflow and sync package version in uv.lock
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Update release preflight script to validate commits against the main branch and remove outdated report file
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Chores

- Uv audit
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Continuous Integration

- Enhance release workflow with preflight checks and reporting
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Documentation

- Add changelog file configuration to pyproject.toml
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Add initial configuration for Zensical documentation site
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Add markdown extensions and update mkdocstrings configuration
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Add missing changelog entries for v1.1.1, v1.1.2, and v1.1.3
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Update README to include 'Why Mother?" section
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Features

- Add release preflight workflow and reporting script
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

---

**Detailed Changes**: [v1.1.3...v1.2.0](https://github.com/Bayer-Group/MotherML/compare/v1.1.3...v1.2.0)


## v1.1.3 (2026-08-31)

### Bug Fixes

- Fix use count based fingerprints as default, including TabPFN precision handling and test updates
  ([#75](https://github.com/Bayer-Group/MotherML/pull/75),
  [`74cac2ec`](https://github.com/Bayer-Group/MotherML/commit/74cac2ec0f93fa7f436c5fc6f4f5c8f711ae1a8a))

- (tabpfn) Restore upstream TabPFN precision handling with float32 for all operations


## v1.1.2 (2026-08-25)

### Bug Fixes

- Solving the `max_features` parameter not being correctly passed to the superclass
  ([#40](https://github.com/Bayer-Group/MotherML/pull/40),
  [`4fd6869c`](https://github.com/Bayer-Group/MotherML/commit/4fd6869c3b3218cedf96060bcf4afe84ba70ece5))


## v1.1.1 (2026-08-21)

### Chores

- Update changelog for version 1.1.0 with bug fixes and new features
  ([`8a69fa24`](https://github.com/Bayer-Group/MotherML/commit/8a69fa246b449f257dacd93d0cbbd28b28090532))


## v1.1.0 (2026-08-20)

### Bug Fixes

- (tabpfn) Corrected a bug concerning unsupported BFloat16 for tabpfn. Now force np.float32 for all
  operations. ([#21](https://github.com/Bayer-Group/MotherML/pull/21),
  [`ca45d17`](https://github.com/Bayer-Group/MotherML/commit/ca45d17e86659d1ee00dae5398737982bd7da63a))

### Features

- (tabicl) add MotherML wrappers, embeddings and uncertainty support
  ([#21](https://github.com/Bayer-Group/MotherML/pull/21),
  [`ca45d17`](https://github.com/Bayer-Group/MotherML/commit/ca45d17e86659d1ee00dae5398737982bd7da63a))


## v1.0.6 (2026-08-18)

### Bug Fixes

- Declare scipy explicitly in base dependencies
  ([#66](https://github.com/Bayer-Group/MotherML/pull/66),
  [`5436111`](https://github.com/Bayer-Group/MotherML/commit/5436111089649394145affef55992369f9401203))

- Local TabPFN import drift and declare SciPy explicitly
  ([#66](https://github.com/Bayer-Group/MotherML/pull/66),
  [`5436111`](https://github.com/Bayer-Group/MotherML/commit/5436111089649394145affef55992369f9401203))


## v1.0.5 (2026-08-17)

### Bug Fixes

- Make MVS default boostrap for catboost tuning
  ([#61](https://github.com/Bayer-Group/MotherML/pull/61),
  [`7141902`](https://github.com/Bayer-Group/MotherML/commit/7141902546738c6a7535bba233f53e87d9c7f4ec))

- Make MVS the default bootstrap for catboost tuning
  ([#61](https://github.com/Bayer-Group/MotherML/pull/61),
  [`7141902`](https://github.com/Bayer-Group/MotherML/commit/7141902546738c6a7535bba233f53e87d9c7f4ec))


## v1.0.4 (2026-07-23)

### Bug Fixes

- Correct typo in workflow configuration for error handling
  ([`1a54319`](https://github.com/Bayer-Group/MotherML/commit/1a5431905e32432f1ce261358b6d17b66d7b6c85))


## v1.0.3 (2026-07-21)

### Bug Fixes

- Add missing readme field in project metadata
  ([`3caa87b`](https://github.com/Bayer-Group/MotherML/commit/3caa87b1e10e2b8ef717199fa36e4b42640a118e))

- Update Python version and actions in workflow configuration
  ([`f14ef68`](https://github.com/Bayer-Group/MotherML/commit/f14ef688432e879ad40c4f06a5bc804fffeefc11))

### Chores

- Uv audit
  ([`e6ac96c`](https://github.com/Bayer-Group/MotherML/commit/e6ac96c67294d314fe4442e7f5c1b5702c485909))


## v1.0.2 (2026-07-21)

### Bug Fixes

- Update entry for docs-python-fences hook to use 'uv run poe'
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

### Documentation

- Add CLAUDE.md for project guidance and command usage
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Enhance documentation and add python code validation for markdown files
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Enhance SKILL.md for changelog management and update __init__.py for version handling
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Remove single-source dependency from pyproject.toml and uv.lock
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Update README and examples for improved clarity and navigation
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Update SKILL.md and CHANGELOG.md for improved documentation workflow and versioning
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))


## v1.0.1 (2026-06-15)

### Bug Fixes

- **ml**: Unify predict_uncertainty output schema across all model backends and pipelines
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Add data and total uncertainty outputs to CatboostRegressorMother predictions
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Add knowledge and data uncertainty outputs to RandomForestRegressorMother predictions
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Enforce single target type in CatboostGaussianProcessRegressorMother, update predict()
  and predict_uncertainty() to accept **kwargs
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Harmonize prediction outputs across model backends
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Improve error message for quantile validation in CatboostRegressorMother
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Initialize quantile_array in CatboostRegressorMother predict method
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **notebook**: Correct minor text and formatting issues in prediction interface guide
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **notebook**: Update predict_uncertainty interface examples and add
  CatboostGaussianProcessRegressorMother ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **pipeline**: Add overloads for mother_cv function to enhance type hints
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **pipeline**: Improve mother_cv function to conditionally return estimators
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **pipeline**: Update multi-target prediction fallback to clarify uncertainty handling
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **tests**: Enhance uncertainty predictions in CatboostRegressorMother and
  GaussianProcessRegressorMother tests ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

### Chores

- Added skill to use ai for changelog and docs update
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- Uv audit ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

### Testing

- **models**: Add regression and classification tests for unified predict behavior
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))


## v1.0.0 (2026-04-17)

- Initial Release
