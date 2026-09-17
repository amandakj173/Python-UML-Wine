# DATA WRANGLING

# Import Packages
import pandas as pd
from pandas.core.methods import describe
from scipy.stats import skew
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np

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
# Identify & remove outliers
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

# Cluster Selection
def gmm_bic(estimator, X):
    return -estimator.bic(X)  # Define BIC scoring function

param_grid = {
    "n_components": range(1,10),
    "covariance_type": ["spherical", "tied", "diag", "full"]
}  # Set up parameter grid of 1-9 components w. all covariance types

grid_search = GridSearchCV(
    GaussianMixture(random_state = 42),  # For reproducability
    param_grid = param_grid,
    scoring = gmm_bic
)  # Run grid search
grid_search.fit(wine_scaled)  # Run grid search over scaled data

best_gmm = grid_search.best_estimator_  # Label best model
best_params = grid_search.best_params_  # Label best model parameters
print("Best model parameters:", best_params)
print("Best negative BIC (higher is better):", grid_search.best_score_)

gmm_classification = best_gmm.predict(wine_scaled)  # Get hard cluster assignments

data_wine["GMM_Cluster"] = gmm_classification  # Add cluster labels to dataframe

# VISUALISATION

# Visualisation
cluster_counts = pd.Series(gmm_classification).value_counts().sort_index()
GMMCluster = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Size": cluster_counts.values
})
print("\nCluster sizes (GMM):", GMMCluster)
print(GMMCluster)  # Cluster size table