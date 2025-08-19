"""
Example command line interface for the proteomics analysis pipeline.

This script demonstrates how to combine the loader, preprocessing and
analysis functions contained in this repository.  It accepts a
quantitative proteomics table and a list of intensity column names and
outputs a processed CSV file.  Optionally, group labels can be
provided to perform differential testing and generate a volcano plot.
"""

import argparse
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from proteomics_data_loader import load_maxquant_proteins, filter_proteins_by_valid_intensities
from proteomics_preprocessing import log2_transform, median_normalize, impute_knn
from proteomics_analysis import run_pca, compute_differential_abundance, volcano_plot


def run_pipeline(file_path: str, intensity_cols: List[str], group_labels: List[int] = None, output_prefix: str = 'processed_proteomics'):
    """Run the proteomics processing and analysis pipeline.

    Parameters
    ----------
    file_path : str
        Path to the proteomics table (e.g., proteinGroups.txt).
    intensity_cols : list of str
        Names of columns containing quantitative intensities.
    group_labels : list of int, optional
        List of group assignments (0 or 1) for each column in intensity_cols.
        Required if you wish to perform differential abundance testing.
    output_prefix : str, optional
        Prefix for output files (default 'processed_proteomics').
    """
    print(f'Reading data from {file_path}...')
    df = load_maxquant_proteins(file_path)
    # Filter proteins with too many missing values
    df = filter_proteins_by_valid_intensities(df, intensity_cols)
    # Preprocess
    df = log2_transform(df, intensity_cols)
    df = median_normalize(df, intensity_cols)
    df = impute_knn(df, intensity_cols)
    # Save processed data
    processed_path = f"{output_prefix}.csv"
    df.to_csv(processed_path, index=False)
    print(f'Processed data saved to {processed_path}')
    # PCA
    scores, explained = run_pca(df, intensity_cols)
    print('PCA explained variance ratio:', explained)
    # If group labels are provided, perform differential analysis
    if group_labels is not None:
        if len(group_labels) != len(intensity_cols):
            raise ValueError('Length of group_labels must equal length of intensity_cols')
        log_fc, p_vals, mean_int = compute_differential_abundance(df, intensity_cols, group_labels)
        fig = volcano_plot(log_fc, p_vals)
        volcano_path = f"{output_prefix}_volcano.png"
        fig.savefig(volcano_path, dpi=300, bbox_inches='tight')
        print(f'Volcano plot saved to {volcano_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run proteomics data processing pipeline')
    parser.add_argument('--file', required=True, help='Path to the proteomics table (e.g., proteinGroups.txt)')
    parser.add_argument('--cols', nargs='+', required=True, help='List of intensity column names to analyse')
    parser.add_argument('--groups', nargs='*', type=int, help='Group assignments (0 or 1) for each intensity column')
    parser.add_argument('--output_prefix', default='processed_proteomics', help='Prefix for output files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args.file, args.cols, args.groups, args.output_prefix)
