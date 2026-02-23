# home-credit-project

This is my home credit risk demo project.

# Home Credit Default Risk: Data Preparation 

This repository contains a data preparation pipeline built in R to process and engineer features for home credit risk modeling. The goal is to transform raw application data into a clean, model-ready format while ensuring strict train/test consistency.

## Pipeline Features

### 1. Data Cleaning & EDA Fixes
- **Anomaly Handling:** Fixed the `DAYS_EMPLOYED` anomaly where the value `365243` (approx. 1000 years) was used as a placeholder.
- **Missing Values:** Imputes `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` using training set medians to avoid data leakage.
- **Normalization:** Converted negative time offsets (Birth, Registration, ID Publish) into absolute positive values.

### 2. Feature Engineering
- **Demographics:** Updated Age and Employment duration from days to years.
- **Financial Ratios:** - **Credit-to-Income:** Total loan amount relative to annual income.
  - **Annuity-to-Income:** Monthly debt-to-income burden.
  - **LTV (Loan-to-Value):** Ratio of goods price to credit amount.
- **Predictive Indicators:** Added binary flags for missing data in external sources.

### 3. Data Aggregation
Aggregates supplementary data to the applicant level (`SK_ID_CURR`):
- **Bureau:** Prior credit counts, debt-to-credit ratios, and overdue amounts.
- **Previous Applications:** Total application counts, approval rates, and refusal history.
- **Installments:** Late payment percentages and payment trends.

## How to Use

1. **Source the script** in your R session or Quarto document:
   ```r
   source("data_preparation.R")

# Aggregate supplementary files
aggs <- aggregate_supplementary(bureau, previous_application, installments_payments)

# Clean and Join
clean_data <- clean_application_data(application_train)
final_data <- join_features(clean_data, aggs)

## Update: Modeling (Feb 22nd); 

To evaluate the credit risk for this project, I explored a diverse set of candidate algorithms, including Logistic Regression (both with full and reduced feature sets), Random Forest, and XGBoost. The selection process was driven by a 3-fold cross-validation strategy on a 5,000-row subsample to identify the model with the highest Area Under the Curve (AUC), as accuracy alone was misleading due to the ~8% default rate imbalance. While the baseline Majority Class Classifier provided an AUC of 0.50, the gradient boosting approach of XGBoost consistently outperformed simpler models by capturing non-linear relationships and interactions within the supplementary data from the bureau and installment files. After performing a Randomized Search to optimize hyperparameters like learning rate and tree depth, I selected XGBoost as my final model because it achieved the strongest discriminative power, yielding a cross-validated AUC of 0.762 and a final Kaggle score of 0.69. This model provides the most reliable balance of precision and recall, allowing the business to effectively differentiate between high-risk applicants and reliable borrowers.


