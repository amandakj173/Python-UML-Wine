# Unsupervised Machine Learning - Wine Chemical Composition Clustering

This project applied Gaussian Mixture Models (GMM) to chemical composition data of wines to discover distinct, naturally occuring profiles without relying on predefined labels. The project combined rigorous data preprocessing, hyperparameter tuning via Bayesian Information Criterion (BIC), and component profile analysis to separate soft probabilistic clusters based on multi-dimensional chemical features.

## Description
Classifying wines purely on raw chemical attributes presents an unique challenge where features exhibit varying scales and non-linear interactions. Thus GMM is better suited to the clustering task compared to traditional distance-based methods that assume spherical geometry and hard cluster boundaries, which fail to capture overlapping feature distributions.

The project implements a structure unsupervised learning workflow:
1. Data wrangling: Import packages, set parameters, and import data
2. Data preprocessing: Evaluate skewness to ensure appropriate modelling architecture. Identify and remove outliers using z-score thresholds of +/- 3 standard deviations from mean. Apply z-score feature scaling using StandardScaler to ensure standardisation and prevent high-magnitude features from dominating distance metrics.
3. Cluster selection: Perform grid search across covariance types (spherical, tied, diagonal, and full) and component counts (1 to 9) using BIC scoring to select the optimal model.
4. Clustering: Cluster datapoints based on component selection and extract the best model.
5. Cluster checks: Evaluate model distribution. Interpret cluster centroids to identify distinct wine profiles from chemical signatures.

## Interpretation

### Cluster Selection
Best model parameters: Spherical covariance with 3 components
Optimal negative BIC score: -1089.44

Grid search across hyperparameter combinations identified the optimal model based on balance between model fit and parsimony.

### Cluster Checks
Cluster distribution: Cluster 1 = 66 instances (38.60%), Cluster 2 = 57 instances (33.33%), Cluster 3 = 48 instances (28.07%)

Cluster distribution indicates groups are balanced.

Feature profiles: Cluster 1 = low alcohol (-0.904), proline (-0.722), and color intensity (-0.878), Cluster 2 = high malic acid (1.001) and color intensity (0.953), and low flavanoids (-1.244) and hue (-1.217), Cluster 3 = high alcohol (0.944), proline (1.186), flavanoids (1.014), and total phenols (0.934).

Clusters showed unique profiles which can be used to correspond with distinct wine profiles. Cluster 1 indicates a low intensity and ligher colour wine profile. Cluster 2 indicates a high acidity and darker wine profile with lower phenol content. Cluster 3 indicates a high-body wine profile with higher phenols.

## Next Steps
1. To visualise the GMM by plotting clustering and component centers.
2. To divide the dataset into a train/test split in order to conduct log-likelihood evaluation on GMM performance.
