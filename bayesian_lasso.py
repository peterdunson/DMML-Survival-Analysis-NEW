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
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42)
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

# # Posterior predictions for the training set
# with model:
#     ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

# # Extract posterior predictive samples for the training set
# if "likelihood" in ppc_train.posterior_predictive:
#     # Average over chains and draws
#     y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
    
#     # Ensure y_pred_ppc is a probability array
#     y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)
    
#     # Evaluate the model on the training data
#     try:
#         train_auc = roc_auc_score(y_train, y_pred_ppc_train)
#         print(f"Train AUC: {train_auc}")
#     except ValueError as e:
#         print(f"Error calculating Train AUC: {e}")

# # Posterior predictions for the test set
# with model:
#     # Extract mean values for intercept and beta
#     intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#     beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
    
#     # Calculate logits for test set
#     logits_test = intercept_mean + np.dot(X_test, beta_mean)
    
#     # Calculate probabilities using the sigmoid function
#     prob_test = 1 / (1 + np.exp(-logits_test))
    
#     # Sample from the Bernoulli distribution using these probabilities
#     ppc_test = pm.draw(pm.Bernoulli.dist(p=prob_test), draws=len(X_test))

# # Evaluate the model on the test data
# try:
#     test_auc = roc_auc_score(y_test, prob_test)  # Use prob_test directly
#     print(f"Test AUC: {test_auc}")
# except ValueError as e:
#     print(f"Error calculating Test AUC: {e}")

#___________________________________________________________________________________________________________________

#fixing train auc

# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score

# # Step 1: Generate synthetic data
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42)
# y = (y > np.median(y)).astype(int)  # Ensure binary classification

# # Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Initialize 20-fold cross-validation
# kf = KFold(n_splits=20, shuffle=True, random_state=42)

# train_aucs = []
# test_aucs = []

# # K-fold Cross-Validation
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     # Step 2: Define the PyMC model
#     with pm.Model() as model:
#         # Priors
#         beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
#         intercept = pm.Normal("intercept", mu=0, sigma=1)
        
#         # Likelihood
#         mu = intercept + at.dot(X_train, beta)
#         likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
        
#         # Sample from the posterior
#         trace = pm.sample(1000, tune=2000, return_inferencedata=True)
    
#     # Posterior predictions for the training set
#     with model:
#         ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])
    
#     # Extract posterior predictive samples for the training set
#     y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
#     y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)
#     train_auc = roc_auc_score(y_train, y_pred_ppc_train)
#     train_aucs.append(train_auc)

#     # Posterior predictions for the test set
#     with model:
#         intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#         beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
#         logits_test = intercept_mean + np.dot(X_test, beta_mean)
#         prob_test = 1 / (1 + np.exp(-logits_test))
    
#     test_auc = roc_auc_score(y_test, prob_test)
#     test_aucs.append(test_auc)

# print(f"Train AUCs: {train_aucs}")
# print(f"Test AUCs: {test_aucs}")
# print(f"Mean Train AUC: {np.mean(train_aucs)} ± {np.std(train_aucs)}")
# print(f"Mean Test AUC: {np.mean(test_aucs)} ± {np.std(test_aucs)}")

#___________________________________________________________________________________________________________________

#confusion matrix as well

# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, confusion_matrix

# # Step 1: Generate synthetic data
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42)
# y = (y > np.median(y)).astype(int)  # Ensure binary classification

# # Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Initialize 20-fold cross-validation
# kf = KFold(n_splits=20, shuffle=True, random_state=42)

# train_aucs = []
# test_aucs = []
# train_conf_matrices = []
# test_conf_matrices = []

# # K-fold Cross-Validation
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     # Step 2: Define the PyMC model
#     with pm.Model() as model:
#         # Priors
#         beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
#         intercept = pm.Normal("intercept", mu=0, sigma=1)
        
#         # Likelihood
#         mu = intercept + at.dot(X_train, beta)
#         likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
        
#         # Sample from the posterior
#         trace = pm.sample(1000, tune=2000, return_inferencedata=True)
    
#     # Posterior predictions for the training set
#     with model:
#         ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])
    
#     # Extract posterior predictive samples for the training set
#     y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
#     y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)
#     y_pred_train_labels = (y_pred_ppc_train > 0.5).astype(int)
#     train_auc = roc_auc_score(y_train, y_pred_ppc_train)
#     train_aucs.append(train_auc)
    
#     # Confusion matrix for training set
#     train_conf_matrix = confusion_matrix(y_train, y_pred_train_labels)
#     train_conf_matrices.append(train_conf_matrix)

#     # Posterior predictions for the test set
#     with model:
#         intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#         beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
#         logits_test = intercept_mean + np.dot(X_test, beta_mean)
#         prob_test = 1 / (1 + np.exp(-logits_test))
#         y_pred_test_labels = (prob_test > 0.5).astype(int)
    
#     test_auc = roc_auc_score(y_test, prob_test)
#     test_aucs.append(test_auc)
    
#     # Confusion matrix for test set
#     test_conf_matrix = confusion_matrix(y_test, y_pred_test_labels)
#     test_conf_matrices.append(test_conf_matrix)

# # Print results
# print(f"Train AUCs: {train_aucs}")
# print(f"Test AUCs: {test_aucs}")
# print(f"Mean Train AUC: {np.mean(train_aucs)} ± {np.std(train_aucs)}")
# print(f"Mean Test AUC: {np.mean(test_aucs)} ± {np.std(test_aucs)}")

# # Print confusion matrices for the first few folds as examples
# for i in range(3):
#     print(f"Train Confusion Matrix Fold {i+1}:\n{train_conf_matrices[i]}")
#     print(f"Test Confusion Matrix Fold {i+1}:\n{test_conf_matrices[i]}")

#_________________________________________________________________________________________________________________

# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, confusion_matrix

# # Step 1: Generate synthetic data
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42)
# y = (y > np.median(y)).astype(int)  # Ensure binary classification

# # Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Initialize 100-fold cross-validation
# kf = KFold(n_splits=100, shuffle=True, random_state=42)

# train_aucs = []
# test_aucs = []
# train_conf_matrices = []
# test_conf_matrices = []

# # K-fold Cross-Validation
# for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     # Check for data leakage by verifying there are no overlapping indices
#     if set(train_index) & set(test_index):
#         raise ValueError(f"Data leakage detected in fold {fold}!")

#     print(f"Fold {fold} - Train indices: {train_index}, Test indices: {test_index}")

#     # Step 2: Define the PyMC model within the fold loop to avoid data leakage
#     with pm.Model() as model:
#         # Priors
#         beta = pm.Laplace("beta", mu=0, b=1, shape=X_train.shape[1])
#         intercept = pm.Normal("intercept", mu=0, sigma=1)
        
#         # Likelihood
#         mu = intercept + at.dot(X_train, beta)
#         likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train)
        
#         # Sample from the posterior
#         trace = pm.sample(1000, tune=2000, return_inferencedata=True)
    
#     # Posterior predictions for the training set
#     with model:
#         ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])
    
#     # Extract posterior predictive samples for the training set
#     y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
#     y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)
#     y_pred_train_labels = (y_pred_ppc_train > 0.5).astype(int)
#     train_auc = roc_auc_score(y_train, y_pred_ppc_train)
#     train_aucs.append(train_auc)
    
#     # Confusion matrix for training set
#     train_conf_matrix = confusion_matrix(y_train, y_pred_train_labels)
#     train_conf_matrices.append(train_conf_matrix)

#     # Posterior predictions for the test set
#     with model:
#         intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#         beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
#         logits_test = intercept_mean + np.dot(X_test, beta_mean)
#         prob_test = 1 / (1 + np.exp(-logits_test))
#         y_pred_test_labels = (prob_test > 0.5).astype(int)
    
#     test_auc = roc_auc_score(y_test, prob_test)
#     test_aucs.append(test_auc)
    
#     # Confusion matrix for test set
#     test_conf_matrix = confusion_matrix(y_test, y_pred_test_labels)
#     test_conf_matrices.append(test_conf_matrix)

# # Print results
# print(f"Train AUCs: {train_aucs}")
# print(f"Test AUCs: {test_aucs}")
# print(f"Mean Train AUC: {np.mean(train_aucs)} ± {np.std(train_aucs)}")
# print(f"Mean Test AUC: {np.mean(test_aucs)} ± {np.std(test_aucs)}")

# # Print confusion matrices for all folds
# for i in range(100):
#     print(f"Train Confusion Matrix Fold {i+1}:\n{train_conf_matrices[i]}")
#     print(f"Test Confusion Matrix Fold {i+1}:\n{test_conf_matrices[i]}")

#_________________________________________________________________________________________________________________

import pymc as pm
import pytensor.tensor as at
import arviz as az
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# Generate synthetic data
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42)
y = (y > np.median(y)).astype(int)  # Ensure binary classification

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize 10-fold stratified cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

train_aucs = []
test_aucs = []

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Run the model 100 times
for iteration in range(100):
    print(f"Iteration {iteration + 1}")

    # K-fold Cross-Validation
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE to the training data
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Check for single-class presence in train and test splits
        if len(np.unique(y_train_smote)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Skipping fold {fold} due to single class present in train or test set.")
            continue

        # Define the PyMC model
        with pm.Model() as model:
            beta = pm.Laplace("beta", mu=0, b=1, shape=X_train_smote.shape[1])
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            
            mu = intercept + at.dot(X_train_smote, beta)
            likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train_smote)
            
            trace = pm.sample(1000, tune=2000, return_inferencedata=True)
        
        # Posterior predictions for the training set
        with model:
            ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])
        
        y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
        y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)

        train_auc = roc_auc_score(y_train_smote, y_pred_ppc_train)
        train_aucs.append(train_auc)

        # Posterior predictions for the test set
        with model:
            intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
            beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
            logits_test = intercept_mean + np.dot(X_test, beta_mean)
            prob_test = 1 / (1 + np.exp(-logits_test))
        
        test_auc = roc_auc_score(y_test, prob_test)
        test_aucs.append(test_auc)

# Print results
print(f"Train AUCs: {train_aucs}")
print(f"Test AUCs: {test_aucs}")
print(f"Mean Train AUC: {np.mean(train_aucs)} ± {np.std(train_aucs)}")
print(f"Mean Test AUC: {np.mean(test_aucs)} ± {np.std(test_aucs)}")



