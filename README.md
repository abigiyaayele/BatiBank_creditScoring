# BatiBank_creditScoring
10 academy-week 6  
## Project Summary

As an Analytics Engineer at Bati Bank, I'm tasked with developing a Credit Scoring Model to support a buy-now-pay-later service in partnership with an eCommerce company. The objective is to enable the bank to evaluate potential borrowers and assign them a credit score that reflects their likelihood of defaulting on a loan. This project involves various tasks, from understanding credit risk to building and deploying machine learning models, ensuring compliance with Basel II Capital Accord.

## Introduction

Credit scoring is a crucial process in financial services, quantifying the risk associated with lending to individuals. Traditional credit scoring models rely on historical data and statistical techniques to predict the likelihood of default. The buy-now-pay-later service aims to provide customers with credit to purchase products, necessitating a robust credit scoring system to minimize financial risk. This project encompasses data analysis, feature engineering, model development, and evaluation to create a reliable credit scoring model.

## Objective

1. Define a proxy variable to categorize users as high risk (bad) or low risk (good).
2. Identify observable features that are good predictors of default.
3. Develop a model to assign risk probability for new customers.
4. Create a credit scoring model from risk probability estimates.
5. Predict the optimal loan amount and duration for each customer.


## Methodologies

1. **Exploratory Data Analysis (EDA):**
   - Understand the dataset structure.
   - Compute summary statistics.
   - Visualize numerical and categorical feature distributions.
   - Analyze feature correlations.
   - Identify missing values and outliers.

2. **Feature Engineering:**
   - Create aggregate features (e.g., total transaction amount, average transaction amount).
   - Extract features (e.g., transaction hour, day, month).
   - Encode categorical variables using one-hot and label encoding.
   - Handle missing values through imputation or removal.
   - Normalize/standardize numerical features.

3. **Default Estimator and WoE Binning:**
   - Construct a default estimator using RFMS (Recency, Frequency, Monetary, Seasonality) formalism.
   - Classify users as good or bad based on RFMS scores.
   - Perform Weight of Evidence (WoE) binning.

4. **Modeling:**
   - Split the data into training and testing sets.
   - Select models (e.g., Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Machines).
   - Train and tune models using techniques like Grid Search and Random Search.
   - Evaluate models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

## Task Steps

1. **Task 1 - Understanding Credit Risk:**
   - Study credit risk concepts and definitions.
   - Review key references and guidelines (Basel II Capital Accord, credit risk articles).

2. **Task 2 - Exploratory Data Analysis (EDA):**
   - Load the dataset and examine its structure.
   - Calculate summary statistics for numerical features.
   - Visualize distributions of numerical and categorical features.
   - Perform correlation analysis.
   - Identify and handle missing values.
   - Detect and address outliers.

3. **Task 3 - Feature Engineering:**
   - Aggregate features like total, average, and standard deviation of transaction amounts per customer.
   - Extract temporal features such as transaction hour, day, month, and year.
   - Encode categorical variables using appropriate techniques.
   - Handle missing values through imputation or removal.
   - Normalize and standardize numerical features.

4. **Task 4 - Default Estimator and WoE Binning:**
   - Develop a default estimator based on RFMS formalism.
   - Assign users into good and bad categories.
   - Implement WoE binning to transform categorical variables.

5. **Task 5 - Modeling:**
   - Split the dataset into training and testing sets.
   - Select and train multiple models.
   - Tune hyperparameters to optimize model performance.
   - Evaluate models using relevant metrics to select the best-performing model.


### Deliverables

- **Task 1:** Report on understanding credit risk.
- **Task 2:** Complete EDA and data visualization.
- **Task 3:** Perform feature engineering.
- **Task 4:** Develop default estimator and perform WoE binning.
- **Task 5:** Train and evaluate models

This project leverages advanced data analysis and machine learning techniques to create a robust credit scoring model, enhancing the buy-now-pay-later service's effectiveness in assessing credit risk.

#Author: Abigiya Ayele
