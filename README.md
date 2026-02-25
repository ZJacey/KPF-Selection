# KPF-Selection
# IDMB: Interleaved-Deletion Markov Boundary Discovery

This repository contains the Python implementation of the **IDMB** algorithm, as introduced in the paper:

> **"An Interleaved-Deletion Markov Boundary Discovery Algorithm for Robust Causal Feature Selection in Manufacturing Systems"**

IDMB provides robust causal feature selection for complex manufacturing data, leveraging conditional independence tests to accurately discover Markov Boundaries.

## Repository Structure

- `IDMB.py`: The main script that runs the Interleaved-Deletion Markov Boundary Discovery algorithm. It reads the dataset, performs the feature selection, and evaluates the selected features.
- `condition_independence_test.py`: Contains various conditional independence (CI) tests (e.g., Fisher-Z test for continuous data, G2 test and Chi-square test for discrete data) required by the causal discovery process.
- `SMOTEENN.py`: A data preprocessing module that applies the SMOTEENN (Synthetic Minority Over-sampling Technique and Edited Nearest Neighbours) algorithm from the `imblearn` library to handle imbalanced datasets.
- `SVM.py`: The evaluation module. It performs classification using the selected Markov Boundary features. It includes 10-fold cross-validation and evaluates models (such as Support Vector Machines and Multi-Layer Perceptron) using Accuracy, Precision, Recall, and F1-score.
- `subsets.py`: Utility functions for generating subsets used during the conditional independence testing phase.
- `dataset.csv`: The dataset used for testing the algorithm.

## Prerequisites

To run this code, you need Python 3.x and the following libraries installed:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib xgboost lightgbm
