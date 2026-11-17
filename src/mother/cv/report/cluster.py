try:
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import plotly.express as px
except ImportError as import_error:
    from mother.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("report", import_error)

import logging
import typing

import numpy as np
import pandas as pd
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdFMCS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

module_logger = logging.getLogger(__name__)


def _import_umap():
    """Import umap lazily.

    ``umap-learn`` can pull TensorFlow/Keras into the process on some installs. Loading that
    before Mother's torch-backed model registry (NODE / TabICL / TabPFN) can hang the interpreter,
    so keep umap off the module import path until a plotting helper actually needs it.
    """
    try:
        import umap
    except ImportError as import_error:
        from mother.errors import ExtrasDependencyImportError

        raise ExtrasDependencyImportError("report", import_error) from import_error
    return umap


def update_plot(
    n_neighbors: int, min_dist: float, df: pd.DataFrame, features: pd.DataFrame, cluster_col: str, expt_col: str
) -> None:
    """Update UMAP plot with new parameters.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for UMAP. Must be positive.
    min_dist : float
        Minimum distance parameter for UMAP. Must be between 0 and 1.
    df : pandas.DataFrame
        DataFrame containing cluster labels and experimental values
    features : pandas.DataFrame
        DataFrame containing feature vectors
    cluster_col : str
        Name of column containing cluster labels
    expt_col : str
        Name of column containing experimental values

    Raises
    ------
    ValueError
        If n_neighbors is not positive or min_dist is not between 0 and 1
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must be between 0 and 1")

    scaler: StandardScaler = StandardScaler()
    scaled_features: np.ndarray = scaler.fit_transform(features)

    umap = _import_umap()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    embedding = reducer.fit_transform(scaled_features)

    umap_df: pd.DataFrame = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "Cluster": df[cluster_col],
            "Experimental": df[expt_col],
        }
    )
    umap_df["Cluster"] = umap_df["Cluster"].astype(str)

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="Cluster",
        title="Interactive UMAP Projection",
        labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        hover_data=["Experimental"],
        width=800,
        height=600,
    )
    fig.show()


def plot_interactive_umap(df: pd.DataFrame, features: pd.DataFrame, cluster_col: str, expt_col: str) -> None:
    """
    Create an interactive UMAP plot with adjustable parameters.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the dataset to be visualized. It should contain the cluster_col collecting cluster labels
        and expt_col collecting experimental parameter values.
    features : pandas.DataFrame
        DataFrame containing the feature vectors, preferably used for clustering.
    cluster_col : str
        Column name for the column containing cluster labels.
    expt_col : str
        Column name containing experimentally determined target values.
    """

    n_neighbors_slider = widgets.IntSlider(value=15, min=1, max=200, step=1, description="n_neighbors:")
    min_dist_slider = widgets.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01, description="min_dist:")

    ui = widgets.VBox([n_neighbors_slider, min_dist_slider])
    out = widgets.interactive_output(
        update_plot,
        {
            "n_neighbors": n_neighbors_slider,
            "min_dist": min_dist_slider,
            "df": widgets.fixed(df),
            "features": widgets.fixed(features),
            "cluster_col": widgets.fixed(cluster_col),
            "expt_col": widgets.fixed(expt_col),
        },
    )
    display(ui, out)


def get_short_cluster_summary(cluster_col: str, df: pd.DataFrame, thresholds: list[int]) -> None:
    """
    Generate a comprehensive summary of the clustering results.

    Parameters:
    ----------
    cluster_col : str
        The name of the column containing cluster labels in the DataFrame.
        This column should contain categorical values representing different clusters.
    df : pd.DataFrame
        The DataFrame containing the clustering results.
    thresholds: list[int]
        A list of thresholds indicating the sizes of clusters to check.
        This function will count how many clusters match the specified threshold requirement.
        Must be non-negative integers.

    Raises:
    -------
    ValueError
        If cluster_col is not in df, df contains null values, or thresholds contains negative values.
    TypeError
        If thresholds contains non-integer values.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in the DataFrame")
    if df[cluster_col].isnull().any():
        raise ValueError(f"Column '{cluster_col}' contains null values")

    try:
        if not all(isinstance(t, (int, np.integer)) for t in thresholds):
            raise TypeError("All thresholds must be integers")
        if any(t < 0 for t in thresholds):
            raise ValueError("All thresholds must be non-negative")
    except TypeError:
        raise TypeError("Thresholds must be a list of integers")

    if df.empty:
        module_logger.info("DataFrame is empty - no clusters to analyze")
        return

    cluster_counts: pd.Series = df[cluster_col].value_counts().sort_index()

    for threshold in sorted(thresholds):
        clusters_of_size: int = sum(1 for count in cluster_counts if count == threshold)
        module_logger.info(f"Number of clusters of size {threshold}: {clusters_of_size}")


def get_silhouette(X: np.ndarray, cluster_labels: np.ndarray, metric: str = "jaccard"):
    """
        Calculate and log the average silhouette score for the given clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    cluster_labels : array-like, shape (n_samples,)
        Cluster labels for each sample.
    metric : str, optional, default="jaccard"
        The metric to use for distance computation.
    """

    silhouette = silhouette_score(X, cluster_labels, metric=metric)
    module_logger.info(f"Average Silhouette Score: {silhouette:.4f}")
    return silhouette


def _filter_unique_mols(molecules: typing.List[Chem.Mol]) -> typing.List[Chem.Mol]:
    """
    Filters the input list of molecules to return only unique and valid RDKit molecule objects.

    Parameters
    ----------
    molecules: List[Chem.Mol]
        A list of RDKit molecule objects.

    Returns
    -------
    List[Chem.Mol]
        A list of unique and valid RDKit molecule objects.
    """
    unique_mols: set[str] = set()
    valid_mols: typing.List[Chem.Mol] = []

    for mol in molecules:
        if mol is None:
            module_logger.warning("Encountered None molecule, skipping.")
            continue
        smiles: str = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles not in unique_mols:
            unique_mols.add(smiles)
            valid_mols.append(mol)

    return valid_mols


def _align_to_mcs(input_molecules: typing.List[Chem.Mol]) -> typing.List[Chem.Mol]:
    """
    Aligns the input molecules to their maximum common substructure (MCS) and computes 2D coordinates.
    This function modifies the input molecules in place to ensure they are aligned and have 2D coordinates.

    Parameters:
    ----------
    input_molecules: typing.List[Chem.Mol]
        A list of RDKit molecule objects to be aligned.

    Returns:
    --------
    input_molecules: typing.List[Chem.Mol]
        A list of RDKit molecule objects with computed 2D coordinates and aligned to MCS.
    """
    for mol in input_molecules:
        if mol is not None:
            rdDepictor.Compute2DCoords(mol)

    if len(input_molecules) > 1:
        mcs = rdFMCS.FindMCS(input_molecules, completeRingsOnly=True)
        mcs_mol: typing.Optional[Chem.Mol] = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is not None:
            rdDepictor.Compute2DCoords(mcs_mol)
            for mol in input_molecules:
                rdDepictor.GenerateDepictionMatching2DStructure(mol, mcs_mol, acceptFailure=True)

    return input_molecules


def visualize_scaffolds(molecules: typing.List[Chem.Mol], max_mols: int = 70):
    """
    Visualizes a list of unique molecular scaffolds from a collection of RDKit molecule objects.

    This function generates a grid image of the unique scaffolds derived from the input
    molecules, computes their 2D coordinates, and attempts to align them based on
    their maximum common substructure (MCS).

    Parameters
    ----------
    molecules: List[Chem.Mol]
        A list of RDKit molecule objects to visualize. The function will extract unique
        scaffolds from these molecules for the visualization.

    max_mols: int, optional
        The maximum number of molecules to display in the visualization grid. Default is 70.

    Returns
    -------
    img:
        A grid image containing the 2D representations of the unique scaffolds,
        with optional legends indicating the scaffold numbers.

    Notes
    -----
    - The function will compute 2D coordinates for each scaffold and attempt to align
      them based on their MCS. If MCS alignment fails, it will fall back to basic
      2D coordinates.
    - The function filters out any empty SMILES representations when extracting unique
      scaffolds.

    Example
    -------
    >>> from rdkit import Chem
    >>> molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]
    >>> img = visualize_scaffolds(molecules)
    >>> img.show()
    """

    if not molecules:
        raise ValueError("No valid molecules to visualize")

    unique_scaffolds_mols = _filter_unique_mols(molecules)

    aligned_scaffolds = _align_to_mcs(unique_scaffolds_mols)

    # Limit number of molecules if needed
    if max_mols:
        aligned_scaffolds = aligned_scaffolds[:max_mols]

    legends = [f"Scaffold {n}" for n in range(1, len(aligned_scaffolds) + 1)]

    img = Draw.MolsToGridImage(
        aligned_scaffolds,
        molsPerRow=4,
        subImgSize=(150, 150),
        legends=legends,
    )

    return img


def _contiguous_runs(labels: np.ndarray) -> list[tuple[float, float, typing.Any]]:
    """Collapse consecutive equal (non-NaN) labels into (start, width, value) intervals.

    Sample ``i`` occupies ``[i, i + 1)`` so edges sit between samples and align with
    integer group boundary lines.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return []

    runs: list[tuple[float, float, typing.Any]] = []
    start = 0
    current = labels[0]

    def _is_nan(value: typing.Any) -> bool:
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False

    for i in range(1, n + 1):
        end_of_run = i == n or _is_nan(labels[i]) != _is_nan(current) or (not _is_nan(current) and labels[i] != current)
        if end_of_run:
            if not _is_nan(current):
                runs.append((float(start), float(i - start), current))
            if i < n:
                start = i
                current = labels[i]
    return runs


def _encode_labels_for_colormap(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary labels to dense integers; also return the unique label values."""
    labels = np.asarray(labels)
    uniques, encoded = np.unique(labels, return_inverse=True)
    return encoded.astype(float), uniques


def _categorical_colors(n: int) -> np.ndarray:
    """Distinct RGBA colors for ``n`` categories (no short-colormap recycling)."""
    if n <= 0:
        return np.empty((0, 4))
    if n == 1:
        return np.array([plt.colormaps.get_cmap("tab20")(0)])
    # Evenly spaced hues stay distinguishable for dozens of Tanimoto groups.
    return plt.colormaps.get_cmap("hsv")(np.linspace(0.0, 0.85, n, endpoint=True))


def _draw_label_runs(
    ax,
    labels: np.ndarray,
    y_center: float,
    height: float,
    cmap=None,
    vmin=None,
    vmax=None,
    colors: np.ndarray | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
) -> None:
    """Draw contiguous horizontal bars for a row of labels."""
    for start, width, value in _contiguous_runs(labels):
        value_f = float(value)
        if colors is not None:
            color = colors[int(value_f)]
        elif vmin is not None and vmax is not None:
            normed = (value_f - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            color = cmap(normed)
        else:
            n_colors = getattr(cmap, "N", 12)
            color = cmap((value_f % n_colors) / max(n_colors - 1, 1))
        ax.broken_barh(
            [(start, width)],
            (y_center - height / 2.0, height),
            facecolors=color,
            edgecolors=edgecolor if edgecolor is not None else "none",
            linewidths=linewidth,
        )


def plot_cv_indices(cv, X, y, group, n_splits, ax=None, lw=10, **kwargs):
    """
    Create a visualization of cross-validation behavior, showing how data is split into training and test sets.

    This function creates a detailed visualization that shows:
    1. How samples are split into training/test sets across different CV iterations
    2. The distribution of class labels
    3. The distribution of group labels (if using group-based CV)

    The plot contains multiple rows:
    - One row per CV split showing training (0) and test (1) sample assignments
    - One row showing the class distribution
    - One row showing the group distribution

    Folds and label rows are drawn as contiguous horizontal intervals (not per-sample
    scatter markers), so large datasets and TimeSeriesSplit unused regions do not
    visually overlap. For group-based CV the x-axis is sorted by group so each group
    is one block and is entirely train or entirely test in every fold.

    Parameters
    ----------
    cv : cross-validation generator
        The cross-validation splitting strategy (e.g., KFold, GroupKFold).
    X : array-like of shape (n_samples, n_features)
        The data to fit. Used only to determine the number of samples.
    y : array-like of shape (n_samples,)
        The target variable for coloring the class distribution row.
    group : array-like of shape (n_samples,)
        Group labels for the samples, used for coloring the group distribution row
        and for group-based CV strategies.
    n_splits : int
        Number of splits/folds in the cross-validation.
    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the visualization. Default is the current axes.
    lw : float, optional
        Controls bar height in the plot (legacy name from the scatter-marker API).
        Default is 10.
    **kwargs : dict, optional
        Additional keyword arguments to pass to matplotlib plotting functions.
        Supported arguments:
        - fontsize: int, size of the title font (default: 15)
        - title: str, plot title (default: CV class name)
        - sort_by_group: bool, sort the x-axis by group for group-based CV
          (default: True). Set False to keep original sample order.
        - max_group_boundaries: int, draw vertical lines between groups only if there
          are at most this many groups (default: 60). With more groups the lines cover
          the plot; the outlined group row still separates the groups.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot, showing CV splits, class distribution,
        and group distribution.

    Notes
    -----
    - The CV splits are shown using a coolwarm colormap (blue for training, red for test)
    - Group row uses one distinct color per group (outlined); class uses viridis on ``y``
    - Contiguous runs of the same assignment share one bar segment
    """
    if ax is None:
        _, ax = plt.subplots()

    n_samples = len(X)
    y_arr = np.asarray(y).reshape(-1)
    group_arr = np.asarray(group).reshape(-1)

    if len(group_arr) != n_samples:
        raise ValueError(f"group length ({len(group_arr)}) must match number of samples in X ({n_samples}).")
    if len(y_arr) != n_samples:
        raise ValueError(f"y length ({len(y_arr)}) must match number of samples in X ({n_samples}).")

    cmap_cv = plt.colormaps.get_cmap("coolwarm")
    cmap_y = plt.colormaps.get_cmap("viridis")
    # Bar height in data coordinates; keep visual weight similar to the old lw=10 markers.
    height = float(np.clip(lw / 15.0, 0.35, 0.85))

    use_groups = "Group" in type(cv).__name__
    groups_for_split = group_arr if use_groups else None
    sort_by_group = bool(kwargs.get("sort_by_group", use_groups))

    # Stable sort by group so each group is one x-block and fold colors stay solid per group.
    if sort_by_group:
        order = np.argsort(group_arr, kind="stable")
    else:
        order = np.arange(n_samples)

    group_plot = group_arr[order]
    y_plot = y_arr[order]
    # Map original index -> plot column
    inv_order = np.empty(n_samples, dtype=int)
    inv_order[order] = np.arange(n_samples)

    ii = -1
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y_arr, groups=groups_for_split)):
        tr = np.asarray(tr, dtype=int)
        tt = np.asarray(tt, dtype=int)
        if use_groups:
            split_groups_train = set(group_arr[tr].tolist())
            split_groups_test = set(group_arr[tt].tolist())
            overlap = split_groups_train & split_groups_test
            if overlap:
                raise ValueError(
                    f"{type(cv).__name__} split {ii} places group(s) {sorted(overlap)[:5]} "
                    "in both train and test; groups were not passed correctly."
                )

        indices = np.full(n_samples, np.nan, dtype=float)
        indices[inv_order[tt]] = 1.0
        indices[inv_order[tr]] = 0.0
        _draw_label_runs(ax, indices, y_center=ii + 0.5, height=height, cmap=cmap_cv, vmin=-0.2, vmax=1.2)

    # Class row: continuous target / labels on a continuous map (avoids one color per float).
    y_min = float(np.min(y_plot))
    y_max = float(np.max(y_plot))
    _draw_label_runs(
        ax,
        y_plot.astype(float),
        y_center=ii + 1.5,
        height=height,
        cmap=cmap_y,
        vmin=y_min,
        vmax=y_max if y_max != y_min else y_min + 1.0,
    )

    group_encoded, _ = _encode_labels_for_colormap(group_plot)
    group_colors = _categorical_colors(int(group_encoded.max()) + 1 if len(group_encoded) else 0)
    _draw_label_runs(
        ax,
        group_encoded,
        y_center=ii + 2.5,
        height=height,
        colors=group_colors,
        edgecolor="0.2",
        linewidth=0.4,
    )

    # Boundaries between consecutive groups on the (possibly sorted) x-axis
    if len(np.unique(group_arr)) <= kwargs.get("max_group_boundaries", 60):
        for i in range(1, n_samples):
            if group_plot[i] != group_plot[i - 1]:
                ax.axvline(i, color="0.1", lw=0.9, zorder=3)

    ax.grid(False)
    xlabel = "Sample index (sorted by group)" if sort_by_group else "Sample index"
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel=xlabel,
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, n_samples],
    )
    title = kwargs.get("title", type(cv).__name__)
    if use_groups:
        n_groups = len(np.unique(group_arr))
        title = f"{title}  ({n_groups} groups; each group entirely in train or test)"
    ax.set_title(title, fontsize=kwargs.get("fontsize", 15))
    return ax
