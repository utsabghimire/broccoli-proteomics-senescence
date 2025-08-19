"""
Functions for loading quantitative proteomics data tables.

This module currently implements a simple loader for MaxQuant
`proteinGroups.txt` output.  The loader removes decoy and contaminant
proteins based on the 'Reverse' and 'Potential contaminant' columns and
returns a cleaned pandas DataFrame.  You can extend these functions to
support other proteomics software outputs by adding appropriate parsers.
"""

import pandas as pd
from typing import List, Optional


def load_maxquant_proteins(file_path: str, sep: str = '\t') -> pd.DataFrame:
    """Load a MaxQuant `proteinGroups.txt` file.

    Parameters
    ----------
    file_path : str
        Path to the MaxQuant proteinGroups.txt file.
    sep : str, optional
        Separator used in the file (default is tab).  Use comma for CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame containing filtered proteins with original columns.

    Notes
    -----
    This function removes rows where the 'Reverse' column is '+' (decoys)
    or the 'Potential contaminant' column is '+' (contaminants).  If
    either column is not present in the file, no filtering is performed
    for that column.
    """
    df = pd.read_csv(file_path, sep=sep)
    # Drop decoy proteins
    if 'Reverse' in df.columns:
        df = df[df['Reverse'] != '+']
    # Drop contaminants
    if 'Potential contaminant' in df.columns:
        df = df[df['Potential contaminant'] != '+']
    return df


def filter_proteins_by_valid_intensities(df: pd.DataFrame, intensity_cols: List[str], min_valid_values: int = 3) -> pd.DataFrame:
    """Remove proteins with too many missing values.

    A protein is retained if it has at least `min_valid_values` non‑NA
    values across the intensity columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing protein intensities.
    intensity_cols : list of str
        Column names corresponding to quantitative intensities.
    min_valid_values : int, optional
        Minimum number of non‑missing values required (default is 3).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    mask = df[intensity_cols].notna().sum(axis=1) >= min_valid_values
    return df.loc[mask].copy()
