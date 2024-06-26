import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from divideEtImpera import divideEtImpera
import tensorflow as tf

# Load the dataset covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

# Normalize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
model = divideEtImpera(X_train, y_train)

# Train the model
model.fit(iterations=8, leaky=0.2, leakyTolerance=0.1)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)