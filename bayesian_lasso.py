#python bayesian_lasso.py

import pymc as pm
import pytensor.tensor as at
import arviz as az
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Step 1: Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
y = (y > np.median(y)).astype(int)  # Ensure binary classification

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the PyMC model
with pm.Model() as model:
    # Priors
    beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    
    # Likelihood
    mu = intercept + at.dot(X_train, beta)
    likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
    
    # Sample from the posterior
    trace = pm.sample(1000, tune=2000, return_inferencedata=True)

# Step 3: Posterior Analysis
# Summary of the posterior
summary = az.summary(trace)
print(summary)

# Posterior predictions
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

# Check the keys in ppc to ensure we are accessing the correct data
print(f"Keys in ppc: {ppc.posterior_predictive.keys()}")

# Extract posterior predictive samples
if "likelihood" in ppc.posterior_predictive:
    y_pred_ppc = ppc.posterior_predictive["likelihood"].mean(axis=0).values.flatten()
    
    # Ensure y_pred_ppc is a probability array
    y_pred_ppc = np.clip(y_pred_ppc, 0, 1)
    
    # Evaluate the model
    try:
        auc = roc_auc_score(y_train, y_pred_ppc)
        print(f"AUC: {auc}")
    except ValueError as e:
        print(f"Error calculating AUC: {e}")
else:
    print("Key 'likelihood' not found in posterior predictive samples.")
    print(f"Available keys in posterior_predictive: {ppc.posterior_predictive.keys()}")
