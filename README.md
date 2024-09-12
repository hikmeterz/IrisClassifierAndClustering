# Iris Classifier and Clustering

This repository contains code and reports for classifying and clustering the Iris dataset using a Decision Tree classifier and a K-Means clustering algorithm.

## Files

1. **kmeans.py**:  
   This file implements a custom K-Means clustering algorithm to group the Iris dataset into clusters. It includes methods for fitting the data and calculating the optimal number of clusters using the elbow method.

2. **report.ipynb**:  
   A Jupyter notebook providing an Exploratory Data Analysis (EDA) of the Iris dataset, followed by the implementation of both Decision Tree and K-Means models. It includes model training, evaluation, and visualizations such as confusion matrices and ROC curves.

3. **report.pdf**:  
   The detailed report summarizing the analysis and results of the project, including comparisons between Decision Tree and K-Means methods, accuracy scores, F1-scores, and visual representations of the results.

## Overview

The Iris dataset is analyzed and processed using:
- Exploratory Data Analysis (EDA) techniques to visualize feature relationships and clean the data.
- Decision Tree classifier for supervised classification of the Iris species.
- K-Means clustering for unsupervised clustering to group the Iris data.

### Main Goals:
- Classify the Iris species with high accuracy using Decision Tree.
- Cluster the Iris dataset into optimal groups using K-Means and validate the results through elbow method and confusion matrices.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
