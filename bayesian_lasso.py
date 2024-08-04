#python bayesian_lasso.py


#___________________________________________________________________________________________________________________

# import pymc as pm
# import pytensor.tensor as at
# import arviz as az
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score
# from imblearn.over_sampling import SMOTE
# from collections import defaultdict

# # Load and preprocess data
# data = pd.read_csv("RF_imputation_NEW.csv")
# data.drop(columns=['deathtime', 'survival_time', 'LOS', 'Unnamed_0', 'V1', 'admittime', 'ID', 'group', 'tLOS', 'subject_id'], inplace=True)
# data['outcome'] = data['outcome'].astype(int)
# predictor_names = data.columns.difference(['outcome'])

# # Separate predictors and target variable
# X = data[predictor_names].values
# y = data['outcome'].values

# # Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Initialize 10-fold stratified cross-validation
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# train_aucs = []
# test_aucs = []

# # Initialize SMOTE
# smote = SMOTE(random_state=42)

# # Feature importance dictionary
# feature_importance_dict = defaultdict(list)

# # Number of iterations
# num_iterations = 10

# for iteration in range(num_iterations):
#     print(f"Iteration {iteration + 1}")
#     iteration_train_aucs = []
#     iteration_test_aucs = []

#     # K-fold Cross-Validation
#     for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Apply SMOTE to the training data
#         X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#         # Check for single-class presence in train and test splits
#         if len(np.unique(y_train_smote)) < 2 or len(np.unique(y_test)) < 2:
#             print(f"Skipping fold {fold} due to single class present in train or test set.")
#             continue

#         # Define the PyMC model
#         with pm.Model() as model:
#             beta = pm.Laplace("beta", mu=0, b=1, shape=X_train_smote.shape[1])
#             intercept = pm.Normal("intercept", mu=0, sigma=1)

#             mu = intercept + at.dot(X_train_smote, beta)
#             likelihood = pm.Bernoulli("likelihood", logit_p=mu, observed=y_train_smote)

#             trace = pm.sample(1000, tune=2000, return_inferencedata=True)

#         # Posterior predictions for the training set
#         with model:
#             ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

#         y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
#         y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)

#         train_auc = roc_auc_score(y_train_smote, y_pred_ppc_train)
#         iteration_train_aucs.append(train_auc)

#         # Posterior predictions for the test set
#         with model:
#             intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
#             beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
#             logits_test = intercept_mean + np.dot(X_test, beta_mean)
#             prob_test = 1 / (1 + np.exp(-logits_test))

#         test_auc = roc_auc_score(y_test, prob_test)
#         iteration_test_aucs.append(test_auc)

#         # Collect feature importances
#         feature_importances = np.abs(beta_mean)
#         for i, importance in enumerate(feature_importances):
#             feature_importance_dict[predictor_names[i]].append(importance)

#     train_aucs.extend(iteration_train_aucs)
#     test_aucs.extend(iteration_test_aucs)

# # Aggregate feature importances
# feature_importances_aggregated = {feature: np.mean(importances) for feature, importances in feature_importance_dict.items()}

# # Sort feature importances
# sorted_feature_importances = sorted(feature_importances_aggregated.items(), key=lambda item: item[1], reverse=True)

# # Print results
# print(f"Train AUCs: {train_aucs}")
# print(f"Test AUCs: {test_aucs}")
# print(f"Mean Train AUC: {np.mean(train_aucs)} ± {np.std(train_aucs)}")
# print(f"Mean Test AUC: {np.mean(test_aucs)} ± {np.std(test_aucs)}")

# # Print sorted feature importances
# print("\nFeature Importances:")
# for feature, importance in sorted_feature_importances:
#     print(f"{feature}: {importance}")

#___________________________________________________________________________________________________________________

import pymc as pm
import pytensor.tensor as at
import arviz as az
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import defaultdict

# Load and preprocess data
data = pd.read_csv("RF_imputation_NEW.csv")
data.drop(columns=['deathtime', 'survival_time', 'LOS', 'Unnamed_0', 'V1', 'admittime', 'ID', 'group', 'tLOS', 'subject_id'], inplace=True)
data['outcome'] = data['outcome'].astype(int)
predictor_names = data.columns.difference(['outcome'])

# Separate predictors and target variable
X = data[predictor_names].values
y = data['outcome'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize 2-fold stratified cross-validation
kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

train_aucs = []
test_aucs = []

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Feature importance dictionary
feature_importance_dict = defaultdict(list)

# Number of iterations
num_iterations = 1

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}")
    iteration_train_aucs = []
    iteration_test_aucs = []

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

            # Reduce sampling and tuning steps for faster execution
            trace = pm.sample(500, tune=1000, return_inferencedata=True)

        # Posterior predictions for the training set
        with model:
            ppc_train = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

        y_pred_ppc_train = ppc_train.posterior_predictive["likelihood"].mean(dim=("chain", "draw")).values
        y_pred_ppc_train = np.clip(y_pred_ppc_train, 0, 1)

        train_auc = roc_auc_score(y_train_smote, y_pred_ppc_train)
        iteration_train_aucs.append(train_auc)

        # Posterior predictions for the test set
        with model:
            intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values
            beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
            logits_test = intercept_mean + np.dot(X_test, beta_mean)
            prob_test = 1 / (1 + np.exp(-logits_test))

        test_auc = roc_auc_score(y_test, prob_test)
        iteration_test_aucs.append(test_auc)

        # Collect feature importances
        feature_importances = np.abs(beta_mean)
        for i, importance in enumerate(feature_importances):
            feature_importance_dict[predictor_names[i]].append(importance)

    train_aucs.extend(iteration_train_aucs)
    test_aucs.extend(iteration_test_aucs)

# Aggregate feature importances
feature_importances_aggregated = {feature: np.mean(importances) for feature, importances in feature_importance_dict.items()}

# Sort feature importances
sorted_feature_importances = sorted(feature_importances_aggregated.items(), key=lambda item: item[1], reverse=True)

# Print results
print(f"Train AUCs: {train_aucs}")
print(f"Test AUCs: {test_aucs}")
print(f"Mean Train AUC: {np.mean(train_aucs)} ± {np.std(train_aucs)}")
print(f"Mean Test AUC: {np.mean(test_aucs)} ± {np.std(test_aucs)}")

# # Print sorted feature importances
# print("\nFeature Importances:")
# for feature, importance in sorted_feature_importances:
#     print(f"{feature}: {importance}")












