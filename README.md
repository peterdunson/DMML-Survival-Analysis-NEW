# Predictive Survival Analysis Proposal

## Aims

The primary aim of this project is to fit a model to predict in-hospital mortality among patients in the heart failure dataset.

## Research Questions

- What are the predictors of in-hospital mortality among ICU-admitted heart failure patients?
- How do demographics, vital signs, comorbidities, and laboratory results influence the risk of hospital mortality?
- How can we ensure the interpretability of our model while also maintaining prediction accuracy?
- Which demographic factors, vital signs, comorbidities, and laboratory results are the most significant predictors of in-hospital mortality among ICU-admitted heart failure patients in the MIMIC-III database?

## Why Does This Matter?

Identifying high-risk heart failure (HF) patients can help clinicians prioritize interventions and improve patient outcomes. Through this data, we aim to enhance our understanding of machine learning models while performing meaningful, substantive healthcare analysis.

## Methods

We will utilize the heart failure dataset extracted from the MIMIC-III database, which includes demographic information, admission details, comorbidities, vital signs, and lab results. The outcome of interest is in-hospital mortality, a binary variable where 1 indicates death and 0 indicates survival. Additionally, we plan to implement the death time variable as a continuous outcome variable tracking when a patient dies while in the ICU.

## First Steps

1. **Exploratory Data Analysis (EDA)**:
   - Summarize the dataset using descriptive statistics and visualizations to understand the distribution of variables and their relationship with the outcome variable.
   
2. **Variable Selection**:
   - Perform univariate analysis to assess the association between each covariate and the outcome using methods like ANOVA, chi-square test, stepwise selection, and goodness-of-fit tests (F-statistic, AIC, BIC, adjusted R-Square).

## Model Development

1. **Initial Models**:
   - Start with logistic regression and Classification and Regression Trees (CART) for the binary outcome.
   - For logistic regression, check for multicollinearity using Variance Inflation Factor (VIF).
   - Include interaction terms, polynomial features, or scaling variables to better capture patterns in the data.
   
2. **Advanced Models**:
   - Implement advanced models like random forest, Cox proportional hazards with time-varying coefficients, random survival forests, hazard function, cumulative hazard function, and survival function with the addition of the death time variable.

3. **Model Evaluation**:
   - Use metrics like the C-index for evaluating survival analysis models.

## Authors

- Ephrata
- Peter
- Nate

## Repository Structure

- `data/`: Contains the heart failure dataset from the MIMIC-III database.
- `scripts/`: Scripts for data cleaning, EDA, and model development.
- `notebooks/`: Jupyter notebooks for exploratory analysis and model building.
- `results/`: Outputs, model results, and evaluation metrics.

## How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Contact

For any questions or suggestions, please contact us via GitHub issues.

---