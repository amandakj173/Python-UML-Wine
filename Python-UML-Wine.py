# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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

scaled_ds = pd.DataFrame(scaled_ds)


# MODELLING

# Identify BIC
def gmm_bic(estimator, x):
    return -estimator.bic(x)

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
print("Best model parameters:", grid_search.best_params_)
print("Best negative BIC:", grid_search.best_score_)

# Clustering
scaled_ds["GMM_Cluster"] = best_gmm.predict(scaled_ds)


# CHECK CLUSTERING

# Check balances & sizes
print(scaled_ds["GMM_Cluster"].value_counts())
print(scaled_ds["GMM_Cluster"].value_counts(normalize=True) * 100)

# Feature Profiles
cluster_means = scaled_ds.groupby("GMM_Cluster").mean()
print(cluster_means)