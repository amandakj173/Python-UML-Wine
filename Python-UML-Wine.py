# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
from pandas.core.methods import describe
from scipy import stats
from scipy.stats import skew
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 2000)

# Import Data
ds = pd.read_csv("wine.csv")

# Basic Data Exploration
print(ds)
ds.info()

# Advanced Data Exploration
print(ds.describe())
print(skew(ds))


# PREPROCESSING

# Outliers
z = np.abs(stats.zscore(ds))
print(z)

threshold_z = 3
outlier_indices = np.where(z > threshold_z)[0]
no_outlier_ds = ds.drop(outlier_indices)
print(no_outlier_ds)

# Scaling
scale = StandardScaler()
scaled_ds = scale.fit_transform(no_outlier_ds)
print(scaled_ds)


# MODELLING

# Identify BIC
def gmm_bic(estimator, X):
    return -estimator.bic(X)

param_grid = {
    "n_components": range(1,10),
    "covariance_type": ["spherical", "tied", "diag", "full"]
}

grid_search = GridSearchCV(
    GaussianMixture(random_state = 42),
    param_grid = param_grid,
    scoring = gmm_bic
)
grid_search.fit(scaled_ds)

best_gmm = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params)
print("Best negative BIC:", grid_search.best_score_)

# Clustering
ds["GMM_Cluster"] = best_gmm.predict(scaled_ds)


# VISUALISATION

# Visualisation
cluster_counts = pd.Series(gmm_classification).value_counts().sort_index()
GMMCluster = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Size": cluster_counts.values
})
print("\nCluster sizes (GMM):", GMMCluster)
print(GMMCluster)  # Cluster size table