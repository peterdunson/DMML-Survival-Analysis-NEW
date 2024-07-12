#python bayesian_lasso.py

# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score

# # Step 1: Generate synthetic data
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# y = (y > np.median(y)).astype(int)  # Ensure binary classification

# # Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 2: Define the PyMC model
# with pm.Model() as model:
#     # Priors
#     beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
#     intercept = pm.Normal("intercept", mu=0, sigma=1)
    
#     # Likelihood
#     mu = intercept + at.dot(X_train, beta)
#     likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
    
#     # Sample from the posterior
#     trace = pm.sample(1000, tune=2000, return_inferencedata=True)

# # Step 3: Posterior Analysis
# # Summary of the posterior
# summary = az.summary(trace)
# print(summary)

# # Posterior predictions
# with model:
#     ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

# # Check the keys in ppc to ensure we are accessing the correct data
# print(f"Keys in ppc: {ppc.posterior_predictive.keys()}")

# # Extract posterior predictive samples
# if "likelihood" in ppc.posterior_predictive:
#     # Average over chains and draws
#     y_pred_ppc = ppc.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
    
#     # Flatten the array to match the shape of y_train
#     y_pred_ppc = y_pred_ppc.flatten()
    
#     # Ensure y_pred_ppc is a probability array
#     y_pred_ppc = np.clip(y_pred_ppc, 0, 1)
    
#     # Evaluate the model
#     try:
#         auc = roc_auc_score(y_train, y_pred_ppc)
#         print(f"AUC: {auc}")
#     except ValueError as e:
#         print(f"Error calculating AUC: {e}")
# else:
#     print("Key 'likelihood' not found in posterior predictive samples.")
#     print(f"Available keys in posterior_predictive: {ppc.posterior_predictive.keys()}")

#___________________________________________________________________________________________________________________

# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score

# # Step 1: Generate synthetic data
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# y = (y > np.median(y)).astype(int)  # Ensure binary classification

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features based on the training data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)  # Apply the same transformation to the test data

# # Step 2: Define the PyMC model
# with pm.Model() as model:
#     # Priors
#     beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
#     intercept = pm.Normal("intercept", mu=0, sigma=1)
    
#     # Likelihood
#     mu = intercept + at.dot(X_train, beta)
#     likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
    
#     # Sample from the posterior
#     trace = pm.sample(1000, tune=2000, return_inferencedata=True)

# # Step 3: Posterior Analysis
# # Summary of the posterior
# summary = az.summary(trace)
# print(summary)

# # Posterior predictions
# with model:
#     ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

# # Check the keys in ppc to ensure we are accessing the correct data
# print(f"Keys in ppc: {ppc.posterior_predictive.keys()}")

# # Extract posterior predictive samples
# if "likelihood" in ppc.posterior_predictive:
#     # Average over chains and draws
#     y_pred_ppc = ppc.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
    
#     # Flatten the array to match the shape of y_train
#     y_pred_ppc = y_pred_ppc.flatten()
    
#     # Ensure y_pred_ppc is a probability array
#     y_pred_ppc = np.clip(y_pred_ppc, 0, 1)
    
#     # Evaluate the model on the test set
#     with model:
#         # Recompute the posterior predictive mean for test data
#         beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
#         intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#         test_mu = intercept_mean + np.dot(X_test, beta_mean)
#         y_pred_test = 1 / (1 + np.exp(-test_mu))  # Convert log-odds to probabilities
    
#     auc = roc_auc_score(y_test, y_pred_test)
#     print(f"AUC: {auc}")
# else:
#     print("Key 'likelihood' not found in posterior predictive samples.")
#     print(f"Available keys in posterior_predictive: {ppc.posterior_predictive.keys()}")

#___________________________________________________________________________________________________________________

#testing extra, data leakage
# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score

# # Step 1: Generate synthetic data with increased complexity
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=10, n_redundant=10, random_state=42)
# y = (y > np.median(y)).astype(int)  # Ensure binary classification

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features based on the training data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)  # Apply the same transformation to the test data

# # Step 2: Define the PyMC model
# with pm.Model() as model:
#     # Priors
#     beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
#     intercept = pm.Normal("intercept", mu=0, sigma=1)
    
#     # Likelihood
#     mu = intercept + at.dot(X_train, beta)
#     likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
    
#     # Sample from the posterior
#     trace = pm.sample(1000, tune=2000, return_inferencedata=True)

# # Step 3: Posterior Analysis
# # Summary of the posterior
# summary = az.summary(trace)
# print(summary)

# # Posterior predictions
# with model:
#     ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

# # Check the keys in ppc to ensure we are accessing the correct data
# print(f"Keys in ppc: {ppc.posterior_predictive.keys()}")

# # Extract posterior predictive samples
# if "likelihood" in ppc.posterior_predictive:
#     # Average over chains and draws
#     y_pred_ppc = ppc.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
    
#     # Flatten the array to match the shape of y_train
#     y_pred_ppc = y_pred_ppc.flatten()
    
#     # Ensure y_pred_ppc is a probability array
#     y_pred_ppc = np.clip(y_pred_ppc, 0, 1)
    
#     # Evaluate the model on the test set
#     with model:
#         # Recompute the posterior predictive mean for test data
#         beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
#         intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#         test_mu = intercept_mean + np.dot(X_test, beta_mean)
#         y_pred_test = 1 / (1 + np.exp(-test_mu))  # Convert log-odds to probabilities
    
#     auc = roc_auc_score(y_test, y_pred_test)
#     print(f"AUC: {auc}")
# else:
#     print("Key 'likelihood' not found in posterior predictive samples.")
#     print(f"Available keys in posterior_predictive: {ppc.posterior_predictive.keys()}")

# # Perform cross-validation to get a more reliable performance estimate
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = []

# for train_idx, test_idx in cv.split(X, y):
#     X_train_cv, X_test_cv = X[train_idx], X[test_idx]
#     y_train_cv, y_test_cv = y[train_idx], y[test_idx]
    
#     # Standardize the features based on the training data
#     X_train_cv = scaler.fit_transform(X_train_cv)
#     X_test_cv = scaler.transform(X_test_cv)
    
#     with pm.Model() as model_cv:
#         # Priors
#         beta = pm.Laplace("beta", mu=0, b=1, shape=X_train_cv.shape[1])
#         intercept = pm.Normal("intercept", mu=0, sigma=1)
        
#         # Likelihood
#         mu = intercept + at.dot(X_train_cv, beta)
#         likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train_cv)
        
#         # Sample from the posterior
#         trace_cv = pm.sample(1000, tune=2000, return_inferencedata=True)
    
#     with model_cv:
#         beta_mean_cv = trace_cv.posterior["beta"].mean(dim=("chain", "draw")).values
#         intercept_mean_cv = trace_cv.posterior["intercept"].mean(dim=("chain", "draw")).values
#         test_mu_cv = intercept_mean_cv + np.dot(X_test_cv, beta_mean_cv)
#         y_pred_test_cv = 1 / (1 + np.exp(-test_mu_cv))  # Convert log-odds to probabilities
    
#     auc_cv = roc_auc_score(y_test_cv, y_pred_test_cv)
#     cv_scores.append(auc_cv)

# print(f"Cross-validated AUC: {np.mean(cv_scores)} ± {np.std(cv_scores)}")

#___________________________________________________________________________________________________________________

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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42)
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

# Extract posterior predictive samples
if "likelihood" in ppc.posterior_predictive:
    # Average over chains and draws
    y_pred_ppc_train = ppc.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
    
    # Ensure y_pred_ppc is a probability array
    y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)
    
    # Evaluate the model on the training data
    try:
        train_auc = roc_auc_score(y_train, y_pred_ppc_train)
        print(f"Train AUC: {train_auc}")
    except ValueError as e:
        print(f"Error calculating Train AUC: {e}")

# Posterior predictions for the test set
with model:
    ppc_test = pm.sample_posterior_predictive(trace, var_names=["likelihood"], samples=len(X_test))

# Extract posterior predictive samples for the test set
if "likelihood" in ppc_test.posterior_predictive:
    # Average over chains and draws
    y_pred_ppc_test = ppc_test.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
    
    # Ensure y_pred_ppc is a probability array
    y_pred_ppc_test = np.clip(y_pred_ppc_test, 0, 1)
    
    # Evaluate the model on the test data
    try:
        test_auc = roc_auc_score(y_test, y_pred_ppc_test)
        print(f"Test AUC: {test_auc}")
    except ValueError as e:
        print(f"Error calculating Test AUC: {e}")
else:
    print("Key 'likelihood' not found in posterior predictive samples for test data.")
    print(f"Available keys in posterior_predictive: {ppc_test.posterior_predictive.keys()}")


