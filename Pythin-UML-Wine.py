# DATA WRANGLING

# Import Packages
import pandas as pd
from scipy.stats import skew
from sklearn import preprocessing
import numpy as np

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