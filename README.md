# Postharvest Senescence Proteomics in Broccoli

This repository contains scripts and instructions for analysing Tandem Mass Tag (TMT) proteomics data from a study on postharvest senescence in broccoli. The workflow starts from a quantitative proteomics table (e.g., MaxQuant proteinGroups.txt) and performs log2 transformation, median normalisation, missing value imputation, exploratory analysis (PCA and clustering) and differential abundance testing with volcano plot visualisation. The pipeline is general enough for other TMT proteomics datasets.

## Contents

- **proteomics_data_loader.py** – load and clean proteomics tables, filtering out contaminants and decoys.
- **proteomics_preprocessing.py** – log2-transform, median-normalise and impute missing values using KNN.
- **proteomics_analysis.py** – run PCA and generate volcano plots for differential analysis.
- **run_pipeline.py** – example command-line script tying the above functions together.
- **requirements.txt** – list of Python dependencies.

## Getting Started

1. Clone this repository or download the scripts.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python run_pipeline.py --file proteinGroups.txt --cols Intensity_1 Intensity_2 ... --groups 0 0 1 1` to process your data and perform basic analysis.

See the original paper for experimental details and interpretation of results.
