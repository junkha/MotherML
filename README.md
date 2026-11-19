# Mother-ML - A ML framework that takes care.

Mother is a machine-learning framework for predicting properties of mainly chemical molecules. The major features are:

- 🔬 **SMILES** preprocessing and standardisation
- 💾 Generation of **feature vectors** from molecules: RDKit descriptors, Morgan, MACCS and other RDKit fingerprints, and pretrained CheMeleon embeddings
- 📈 Grouping and cross-validation based on chemical similarity, scaffolds or time
- 💻 Model training with CatBoost, random forest and lasso, Gaussian-process CatBoost, learning-to-rank, the tabular foundation models TabPFN and TabICL, and NODE neural networks, plus feature selection
- 🎯 A common **uncertainty** interface (`predict_uncertainty`) across model backends
- 🚴 Cross-validation and hyperparameter optimisation of complete pipelines with Optuna
- 🌀 Handling of gene expression data from transcriptomics experiments, including different normalisation techniques
- ✨ Explainability with *SHAP* for TabICL models (support for other models will follow)

Mother provides methods for each of these steps in the form of sklearn transformer objects. By that, all methods are designed to be easily accessible and usable in a modular way. The methods can be combined to ML workflows with [sklearn pipelines, column transformers, and feature unions](https://scikit-learn.org/dev/modules/compose.html).

All methods can be used as sklearn `transformer` or `estimator`. Combination with other methods, or own methods and models (e.g. using mother preprocessing with other model) is therefore straightforward. To be as compatible as possible, every transformer can be constructed using a dictionary containing the required parameters. However, to provide some convenience to the users, a settings class [MotherSettings](https://github.com/Bayer-Group/MotherML/blob/main/src/mother/settings.py) is provided. This class can be used to store all relevant settings for your ML project.

📖 **Documentation:** [bayer-group.github.io/MotherML](https://bayer-group.github.io/MotherML/)

## Why Mother?

Mother keeps the parts of a molecular ML workflow connected without requiring you to leave the scikit-learn ecosystem.

- **Tune an entire workflow from one place.** Mother model wrappers define their own Optuna search spaces, and `PipelineWithHyperparameterRooting`, `ColumnTransformerWithHyperparameterRooting`, and `FeatureUnionWithHyperparameterRooting` collect them across nested steps using familiar `step__parameter` names. You can therefore tune preprocessing, feature selection, and the final estimator together while retaining scikit-learn composition.
- **Exchange compatible transformers and estimators.** Preprocessing, molecular feature generation, feature selection, cross-validation grouping, and models follow scikit-learn's estimator and transformer interfaces. Mother components work in `Pipeline`, `ColumnTransformer`, and `FeatureUnion`, and can be combined with your own scikit-learn components.
- **Use consistent, analysis-ready outputs.** Components preserve pandas-friendly tabular data where appropriate. `predict_uncertainty(...)` exposes a common prediction and uncertainty schema across model backends, while `mother_cv(...)` returns fold-level predictions together with evaluation metadata for straightforward comparison and downstream analysis.
- **Evaluate chemical models more realistically.** Chemistry-aware grouping, including Tanimoto-similarity and Murcko-scaffold groups, makes it practical to use group-aware splits that better test generalisation to dissimilar compounds.
- **Choose models without rewriting the workflow.** The model registry discovers integrated model wrappers and presents a common interface for CatBoost, random forest, lasso, TabPFN, TabICL and NODE (when their extras are installed), and compatible custom models.
- **Make workflows reproducible and configurable.** `MotherSettings` keeps input, preprocessing, feature generation, cross-validation, model, and tuning configuration in one validated object that can be loaded from or written to YAML.

## Install

Mother supports Python 3.11 to 3.14.

```bash
uv add mother-ml
# or
pip install mother-ml
```

> [!CAUTION]
> There is also a different `mother` package on PyPI. Be sure to install `mother-ml`.

### Optional Features and Extras

To keep the package size small, some dependencies are added as optional extras. These extras provide additional functionality for specific use cases:

| Extra | Description | Key Packages | Notes |
|-------|-------------|--------------|-------|
| `report` | Visualisation and reporting tools | matplotlib, seaborn, plotly, umap-learn, igraph, leidenalg | For plots, cluster reports and Optuna QC |
| `rna` | Gene expression (transcriptomics) preprocessing | anndata, scanpy | Needed for `mother.ml.rna` |
| `clustering` | K-medoids compound clustering | kmedoids | Needed for `KMedoidsGroupingFromMols` |
| `torch` | PyTorch | torch | **Adds several GB to the environment!** |
| `tabpfn` | TabPFN models | tabpfn, torch | Tabular foundation model; downloads a checkpoint on first use |
| `tabicl` | TabICL models and SHAP explainability | tabicl[shap], torch, matplotlib | Tabular foundation model; downloads a checkpoint on first use |
| `node` | NODE neural networks with uncertainty heads | skorch, zuko, torch | Neural oblivious decision ensembles |
| `chemprop` | CheMeleon / GNN molecular fingerprints | chemprop | Pretrained molecular embeddings |

The extras that pull in torch add several GB to the environment.

#### Installation Examples

**Using pip:**

```bash
# Install with report generation support
pip install 'mother-ml[report]'

# Install with TabICL support (includes torch)
pip install 'mother-ml[tabicl]'

# Install multiple extras
pip install 'mother-ml[report,tabpfn,tabicl,node]'
```

**Using uv:**

```bash
# Install with specific extras
uv add mother-ml --extra report --extra tabicl --extra node
```

## Usage

A basic example can be found in the [example regression notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/02_regression/01_basic_regression.ipynb). Other
examples are in the [examples folder](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks) (see [Examples](#examples) below).

### 🔬 SMILES preprocessing and mol-object generation

SMILES preprocessing is done with the `StandardizerTransformer` class. The class is used to preprocess SMILES strings to construct a pipeline from SMILES to rdkit mol-objects with:

```python
import pandas as pd
from sklearn import pipeline as sklearn_pipeline
from mother.preprocessing.core import SmilesToMolTransformer, StandardizerTransformer

structure_data = pd.DataFrame(
    {
        "smiles": [
            "CCO",
            "CCN",
            "c1ccccc1",
            "CC(=O)O",
            "CC(C)O",
            "CCCC",
        ]
    }
)

preprocessor: sklearn_pipeline.Pipeline = sklearn_pipeline.Pipeline(
    [
        (
            "smiles_standardizer",
            StandardizerTransformer(
                flags=["STANDARDIZE", "DESALT", "NEUTRALIZE"],
                smiles_col="smiles",
            ),
        ),
        ("smiles_to_mol", SmilesToMolTransformer(molecule_col="Molecule")),
        # Add other column transformations here if needed
    ],
    memory=None,
).set_output(transform="pandas")

mol_data: pd.DataFrame = preprocessor.fit_transform(structure_data)

```

Customise by changing the `flags` attribute. All available flags are described in the [standardisation docs](https://bayer-group.github.io/MotherML/standardization/).

### 💾 Feature Generation

Mother provides several feature generators, among them `MaccsFingerprints`, `MorganFingerprints`, and `ChemicalDescriptors`:

```python
from sklearn import pipeline as sklearn_pipeline
from mother.feature_generation.core import (
    ChemicalDescriptors,
    MaccsFingerprints,
    MorganFingerprints,
)

feature_generator = sklearn_pipeline.FeatureUnion(
    transformer_list=[
        ("maccs", MaccsFingerprints()),
        ("morgan", MorganFingerprints()),
        ("desc", ChemicalDescriptors()),
    ],
).set_output(transform="pandas")

features: pd.DataFrame = feature_generator.fit_transform(mol_data["Molecule"])

```

The `FeatureUnion` class is used to combine the feature generators. Each feature generator can be configured.
Morgan and generic fingerprints are count-based by default; set `use_counts=False` for bit vectors.

Other RDKit fingerprints (`RDKitFP`, `MorganFP`, `AtomPairFP`, `TopologicalTorsionFP`) are available through `FingerprintsGeneric`:

```python
from mother.feature_generation import FingerprintsGeneric

atom_pairs = FingerprintsGeneric(fp_type="AtomPairFP", parameters={"fpSize": 2048})
atom_pair_features = atom_pairs.fit_transform(mol_data["Molecule"])
```

Pretrained **CheMeleon** embeddings from a graph neural network are available with the `chemprop` extra via
`CheMeleonFingerprintFactory` / `CheMeleonFingerprintTransformer`; see the
[CheMeleon notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/04_feature_engineering/04_chemeleon_fingerprints.ipynb).

### 📈 Grouping and Cross-Validation

For cross-validation, or test-set selection based on chemical similarity, mother provides transformer classes for
generating groups, for example `TanimotoGroupingFromMols`:

```python
import mother.cv as cv_module

groups_engine = cv_module.TanimotoGroupingFromMols(similarity_threshold=0.3)

groups: pd.DataFrame = groups_engine.set_output(transform="pandas").fit_transform(mol_data["Molecule"])

```

These groups can be used, e.g. in the `GroupKFold` class from the `sklearn.model_selection` module:

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=3)
```

Other grouping strategies:

| Class | Groups molecules by | Typical use |
|-------|---------------------|-------------|
| `TanimotoGroupingFromMols` | Tanimoto similarity clusters (sphere exclusion) | Default for testing generalisation to dissimilar compounds |
| `MurckoScaffoldGroupingFromMols` | Bemis-Murcko scaffold | Testing generalisation to new scaffolds |
| `TimeSeriesGrouping` | Time / date column | Prospective validation (train on the past, test on the future) |
| `HdbscanGroupingFromMols`, `KMedoidsGroupingFromMols` (in `mother.cv.core`) | Density- or medoid-based clusters | Alternative cluster-based splits; k-medoids needs the `clustering` extra |
| `DefaultGrouping` | No grouping (one group per row) | Random splits |

### 💻 Model Training

The standard model setup of Mother consists of a `feature selection`, and a classification- or regression
model. Both are based on `Catboost`. The standard setup for a regression task would be:

```python
import mother.pipeline_utils as mother_takes_care
from mother import ml

model_settings = {
    "feature_selection_flags": ["DROP_CORRELATED", "DROP_CONSTANT", "DROP_DUPLICATES"],
    "correlation_threshold": 0.9,
    "categorical_features": [],
    "feature_selection_type": "catboost",
    "model_type": "regression",
    "target_type": "single_target",
}
pipeline_settings = {
    "remainder": "drop",
    "verbose_feature_names_out": False,
}

model = ml.PipelineWithHyperparameterRooting(
    [
        (
            "feature_selector",
            mother_takes_care.get_feature_selection_pipeline(
                settings=model_settings,
                pipeline_settings=pipeline_settings,
                data=features,
                cv=cv,
            ).set_output(transform="pandas"),
        ),
        (
            "ml_model",
            ml.CatboostRegressorMother(
                target_type="single_target",
                logging_level="Silent",
                random_seed=42,
                iterations=10,
            ),
        ),
    ]
)

targets = pd.Series([0.2, 0.4, 0.7, 1.1, 1.5, 2.0], name="target")
model.fit(features, targets)
```

Here, we use the extended sklearn pipeline `PipelineWithHyperparameterRooting` for some additional methods for hyperparameter
tuning.

Without feature selection, this is simplified:

```python
model = ml.CatboostRegressorMother(target_type="single_target", logging_level="Silent")
```

Any other sklearn model, or own model can be used instead of `CatboostRegressorMother`. An example, on how a custom
preprocessing step is added to the model, can be found in the
[example notebook on custom preprocessing](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/01_basics/03_custom_preprocessing.ipynb).

#### Available models

| Algorithm key | Model class | Unique selling points | Extra |
|---------------|-------------|-----------------------|-------|
| `catboost` | `CatboostRegressorMother` | Strong, fast default for tabular and fingerprint data; handles categorical features and missing values natively; multi-target regression with missing labels (`MultiRMSEWithMissingValues`); data and knowledge uncertainty with `RMSEWithUncertainty` and virtual ensembles | — |
| `catboost` | `CatboostClassifierMother` | Binary, multiclass and multi-label classification; tunes Logloss against Focal loss for imbalanced data; balanced class weights by default; knowledge uncertainty from virtual ensembles (binary, single target) | — |
| `catboost` | `CatboostGaussianProcessRegressorMother` | Gaussian-process-like knowledge uncertainty from an ensemble of posterior samples ([Ustimenko et al.](https://arxiv.org/abs/2206.05608)); scales to large datasets and many or mixed features where an exact Gaussian process becomes impractical | — |
| `catboost` | `CatboostRankerMother` | Learning-to-rank within groups (e.g. series or assays) with YetiRank, PairLogit and other ranking losses; returns ranks and uncertainty; useful when only the order matters or measurements are not comparable across groups | — |
| `randomforest` | `RandomForestRegressorMother` | Robust baseline that needs little tuning; quantile regression forest, so it predicts arbitrary quantiles and prediction intervals without assuming a distribution | — |
| `randomforest` | `RandomForestClassifierMother` | Robust, low-variance classification baseline with balanced class weights; hard to overfit on small datasets | — |
| `lasso` | `LassoRegressorMother`, `LassoClassifierMother` | Sparse linear models: fast, interpretable coefficients and built-in feature selection; good baseline for small datasets with many features (e.g. gene expression) | — |
| `tabpfn` | `TabPFNRegressorMother`, `TabPFNClassifierMother` | Pretrained tabular foundation model: strong accuracy on small datasets (up to about 10,000 samples) with little or no tuning; regression uncertainty from predicted quantiles; embeddings via `TabPFNEmbeddingTransformer` | `tabpfn` |
| `tabicl` | `TabICLRegressorMother`, `TabICLClassifierMother` | Pretrained in-context-learning foundation model that scales to larger tables than TabPFN; SHAP explanations via `explain` and `plot_shap`; regression uncertainty from predicted quantiles; embeddings via `TabICLEmbeddingTransformer` | `tabicl` |
| `node` | `NODERegressor`, `NODEClassifier` | Differentiable decision-tree ensembles (neural oblivious decision ensembles); full predictive distributions with normalising-flow heads; data and knowledge uncertainty via MC dropout (BALD or BALSA-EMD); multi-target, categorical features and learned embeddings | `node` |

The model registry lists what is available in your environment and returns model classes by algorithm and task:

```python
from mother import ml

print(ml.get_available_algorithms())

regressor_class = ml.get_model_class_by_algorithm_and_type("catboost", "regression")
```

Own models can be registered with the `@ml.register_model` decorator; see the
[models documentation](https://bayer-group.github.io/MotherML/mother/models/).

### 🎯 Uncertainty

Models that support it implement `predict_uncertainty`, which returns one table with the same columns for every backend:
`pred`, `mean_predictions`, `knowledge_uncertainty` (epistemic: lack of similar training data),
`data_uncertainty` (aleatoric: noise in the labels) and `total_uncertainty`. Columns a model does not estimate are left empty.

```python
uncertainty_model = ml.CatboostRegressorMother(
    target_type="single_target",
    loss_function="RMSEWithUncertainty",
    logging_level="Silent",
    iterations=100,
)
uncertainty_model.fit(features, targets)
uncertainty = uncertainty_model.predict_uncertainty(features)
```

Which uncertainty a model provides depends on the backend. CatBoost uses virtual ensembles and, with `RMSEWithUncertainty`,
also a data uncertainty. `CatboostGaussianProcessRegressorMother` gives a Gaussian-process knowledge uncertainty. The random
forest regressor is a quantile regression forest, TabPFN and TabICL predict quantiles, and NODE uses MC dropout or a normalising-flow head.
See the [uncertainty interface notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/01_basics/06_predict_predict_uncertainty_interface.ipynb).

### Cross-validation

Having used any sklearn `pipeline`, or sklearn `estimator` or `transformer` classes, we can use the sklearn
methods for e.g. cross-validation (`cross_validate`):

```python
from sklearn.model_selection import cross_validate

cross_validate(model, features, targets, groups=groups.values.ravel(), cv=cv, n_jobs=1)
```

A more convenient method is provided by mother. It returns the out-of-fold predictions, and the uncertainties if the model
supports them, together with the fold and group of each row:

```python
import mother.pipeline_utils as mother_takes_care

mother_takes_care.mother_cv(
    estimator=model,
    X=features,
    y=targets,
    groups=groups,
    cv=cv,
)
```

With `return_estimators=True`, mother_cv can provide all *k* fitted estimators from k-fold cross validation.

```python
import mother.pipeline_utils as mother_takes_care

cv_results, estimator_dict = mother_takes_care.mother_cv(
    estimator=model,
    X=features,
    y=targets,
    groups=groups,
    cv=cv,
    return_estimators=True,
)
```

If the base estimator model is a classifier, returned fitted estimators can be combined as a *CV Ensemble Classifier* in mother.
The ensemble model can be used for predictions, i.e. with `predict_uncertainty()`:

```python
from mother.ml.ensembles.cvEnsemble import EnsembleAverageUncertaintyClassifier

ensemble = EnsembleAverageUncertaintyClassifier(estimators=estimator_dict["estimators"])
ensemble_prediction = ensemble.predict_uncertainty(X=features)
```

### 🚴 Hyperparameter Optimization

The Mother object `MotherTuner` uses optuna to optimize hyperparameters:

```python
import mother.optimization as opt

tuner = opt.MotherTuner(
    scorer="r2",
    n_trials_optuna=2,
    n_threads_optuna=1,
)

model_tuned = tuner.optimize(
    model,
    features,
    targets,
    cv,
    groups=groups.values.ravel(),
)
```

The function `model.get_hyperparameter_space` returns the hyperparameter space for the model. It is implemented for all
Mother models and is collected automatically through `PipelineWithHyperparameterRooting` and the other
`*WithHyperparameterRooting` classes.

For examples, on how to customize the hyperparameter optimization, or define hyperparameters for your own
models, see the [example notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/05_advanced/01_custom_hyperparameter_optimization.ipynb).

### ✨ Explainability

TabICL models provide SHAP values directly (with the `tabicl` extra):

```python notest
from mother.ml.models.m_tabicl import TabICLRegressorMother

tabicl_model = TabICLRegressorMother().fit(features, targets)
tabicl_model.explain(features.iloc[:5], max_evals=2 * features.shape[1] + 1)
tabicl_model.plot_shap(kind="beeswarm")
```

### 🌀 Handling Gene expression data from transcriptomics experiments including different normalisation techniques

The RNA processing pipeline is implemented in the RNA class, which incorporates various preprocessing steps tailored for RNA sequencing data. All RNA code can be found in the rna.py file and needs the `rna` extra.

The pipeline includes normalization, feature selection, and discretization, utilizing the power of the scikit-learn framework. The normalization methods available are "Scanpy," "UQ," "CUF," and "CPM.". You can customise the pipeline to your needs, or try different normalisation
methods and bin sizes in hyper-parameter tuning. The pipeline can be fitted and re-applied to avoid data-leakage during the normalisation.

Here's how to set up and use the RNA processing pipeline:

```python
from mother.ml.rna import RNA
import numpy as np
import pandas as pd

rna_model = RNA(
    n_features=3,  # Number of features (=genes) to keep for the prediction.
    n_bins=20,  # Number of bins to use for the discretisation of the target variable.
    normalisation_method="UQ",  # Which normalisation to use
)

rng = np.random.default_rng(42)
rna_data_train = pd.DataFrame(
    rng.integers(0, 200, size=(20, 8)),
    columns=[f"gene_{i}" for i in range(8)],
)
rna_data_test = pd.DataFrame(
    rng.integers(0, 200, size=(5, 8)),
    columns=rna_data_train.columns,
)
y_train = (rna_data_train["gene_0"] > 100).astype(int).rename("class")

# Fit the pipeline to your RNA sequencing data
transformed_train_data: pd.DataFrame = rna_model.fit_transform(rna_data_train, y_train)
transformed_test_data: pd.DataFrame = rna_model.transform(rna_data_test)

```

A complete walkthrough of the RNA functionality is found in the [example notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/04_feature_engineering/03_rna_preprocessing.ipynb).

## Examples

The [example notebooks](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks) are grouped by topic:

| Folder | Topics |
|--------|--------|
| [`01_basics`](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks/01_basics) | Framework overview, configuration with `MotherSettings`, custom preprocessing, compound clustering, the prediction and uncertainty interface |
| [`02_regression`](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks/02_regression) | CatBoost and random forest regression, regression with uncertainty, multi-task models, TabICL |
| [`03_classification`](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks/03_classification) | Lasso, binary classification with uncertainty, ordinal classification, TabICL |
| [`04_feature_engineering`](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks/04_feature_engineering) | Feature selection, Boruta, RNA preprocessing, CheMeleon fingerprints |
| [`05_advanced`](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks/05_advanced) | Custom hyperparameter optimisation, TabPFN, learning-to-rank, NODE, CheMeleon with CatBoost, TabICL embeddings |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/Bayer-Group/MotherML/blob/main/CONTRIBUTING.md) for the
pull-request workflow and the [development docs](https://bayer-group.github.io/MotherML/development/dev/) for setting up the
environment, running the tests and the conventional-commit messages used for automatic releases.

## Acknowledgements

Thank you to the following contributors:

- Thomas Wolf
- Lukas Hebing
- Kai Sommer

and [all the others](https://github.com/Bayer-Group/MotherML/blob/main/pyproject.toml).
