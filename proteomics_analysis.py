"""
Analysis routines for quantitative proteomics data.

This module provides functions to perform exploratory analyses such as
principal component analysis (PCA) as well as simple differential
abundance testing and volcano plot visualisation.  These functions are
generic and can be adapted to different types of quantitative datasets.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Sequence, Tuple
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def run_pca(df: pd.DataFrame, columns: List[str], n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Run principal component analysis on intensity data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with intensity columns.
    columns : list of str
        Names of columns to include in the PCA.
    n_components : int, optional
        Number of principal components to compute (default is 2).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the PCA scores (samples × components) and the
        proportion of variance explained by each component.
    """
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(df[columns].values)
    explained = pca.explained_variance_ratio_
    return scores, explained


def compute_differential_abundance(df: pd.DataFrame, columns: List[str], group_labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate fold changes and p‑values between two groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with intensity data.
    columns : list of str
        Names of intensity columns (must correspond to group_labels).
    group_labels : sequence of int
        Sequence of zeros and ones indicating group membership for each
        column (length must equal len(columns)).  Only two groups are
        supported.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (log2 fold change, p‑values, mean intensity across samples).
    """
    group1_cols = [col for col, lbl in zip(columns, group_labels) if lbl == 0]
    group2_cols = [col for col, lbl in zip(columns, group_labels) if lbl == 1]
    # Compute log fold change (group2 - group1)
    mean1 = df[group1_cols].mean(axis=1)
    mean2 = df[group2_cols].mean(axis=1)
    log_fc = mean2 - mean1
    # Student's t-test (unequal variance)
    _, p_vals = ttest_ind(df[group1_cols].values, df[group2_cols].values, axis=1, equal_var=False, nan_policy='omit')
    # Mean intensity (useful for filtering)
    mean_intensity = df[columns].mean(axis=1)
    return log_fc.values, p_vals, mean_intensity.values


def volcano_plot(log_fc: np.ndarray, p_vals: np.ndarray, fc_threshold: float = 1.0, p_threshold: float = 0.05, title: str = "Volcano plot") -> plt.Figure:
    """Generate a simple volcano plot.

    Points above the `p_threshold` and beyond the `fc_threshold` in absolute
    value can be highlighted; for simplicity this function colours points
    conditionally.

    Parameters
    ----------
    log_fc : array-like
        Log2 fold changes for each protein.
    p_vals : array-like
        P‑values for each protein.
    fc_threshold : float, optional
        Threshold for absolute log2 fold change (default 1.0).
    p_threshold : float, optional
        Significance threshold for p‑values (default 0.05).
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the volcano plot.
    """
    neg_log_p = -np.log10(p_vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    # Colour points based on thresholds
    colors = []
    for fc, p in zip(log_fc, p_vals):
        if (abs(fc) >= fc_threshold) and (p < p_threshold):
            colors.append('red')  # Significant & large change
        elif p < p_threshold:
            colors.append('orange')  # Significant only
        else:
            colors.append('gray')  # Not significant
    ax.scatter(log_fc, neg_log_p, c=colors, s=10, alpha=0.7)
    ax.axvline(fc_threshold, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(-fc_threshold, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(-np.log10(p_threshold), color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Log₂ fold change (group2 – group1)')
    ax.set_ylabel('–log₁₀(p-value)')
    ax.set_title(title)
    return fig
