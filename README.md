# home-credit-project

This is my home credit risk demo project.

# Home Credit Default Risk: Data Preparation Pipeline

This repository contains a modular data preparation pipeline built in R to process and engineer features for credit risk modeling. The goal is to transform raw application data and supplementary relational data into a clean, model-ready format while ensuring strict train/test consistency.

## 🛠️ Pipeline Features

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

## 🚀 How to Use

1. **Source the script** in your R session or Quarto document:
   ```r
   source("data_preparation.R")

# Aggregate supplementary files
aggs <- aggregate_supplementary(bureau, previous_application, installments_payments)

# Clean and Join
clean_data <- clean_application_data(application_train)
final_data <- join_features(clean_data, aggs)
