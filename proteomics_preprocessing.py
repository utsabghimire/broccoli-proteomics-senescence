"""
Preprocessing functions for quantitative proteomics data.

This module contains utilities to transform intensity values to a
logarithmic scale, normalise across samples and impute missing values.
These steps help stabilise variance and make downstream statistical
tests more robust.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from typing import List


def log2_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log2 transformation to specified columns.

    A pseudocount of 1 is added to avoid taking the log of zero.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing protein intensities.
    columns : list of str
        Column names to transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with log2‑transformed intensity columns.
    """
    df = df.copy()
    df[columns] = np.log2(df[columns] + 1)
    return df


def median_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Perform median normalisation on intensity columns.

    For each sample column, subtract its median value so that the
    distribution of intensities is centred around zero.  This helps
    correct for systematic differences in sample loading or labelling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing protein intensities.
    columns : list of str
        Columns to normalise.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalised intensity columns.
    """
    df = df.copy()
    medians = df[columns].median()
    df[columns] = df[columns] - medians
    return df


def impute_knn(df: pd.DataFrame, columns: List[str], n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using K‑nearest neighbours imputation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing protein intensities.
    columns : list of str
        Columns to impute.
    n_neighbors : int, optional
        Number of neighbours to use (default is 5).

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values in the specified columns.
    """
    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df[columns])
    df[columns] = imputed
    return df
