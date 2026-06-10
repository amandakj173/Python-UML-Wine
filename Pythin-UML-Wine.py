# DATA WRANGLING

# Import Packages
import pandas as pd
from scipy.stats import skew
from sklearn import preprocessing
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 2000)

# Import Data
data_wine = pd.read_csv("wine.csv")

# Initial Data Exploration
print(data_wine)
data_wine.info()

# Advanced Data Exploration
print(data_wine.describe())
skewness_wine = skew(data_wine)
print(skewness_wine)


# PREPROCESSING
scaler = preprocessing.StandardScaler().fit(data_wine)  # Create scaler object & fit to data (find mean & sd)
scaler  # Print representation of object
scaler.mean_  # Look at calculated means
scaler.scale_  # Look at calculated sds
wine_scaled = scaler.transform(data_wine)  # Apply scaling
wine_scaled  # View tranformed array


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