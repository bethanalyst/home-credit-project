# Home Credit Default Risk: Data Pipeline & Predictive Modeling

## Business Problem & Project Objective

Many individuals struggle to get loans due to insufficient or non-existent credit histories. Home Credit strives to broaden financial inclusion by utilizing alternative data to predict clients' repayment abilities. The objective of this project was to build a robust end-to-end data pipeline and machine learning model to predict the probability that an applicant will default on a loan. By identifying high-risk borroIrs more accurately, the business can reduce financial loss while approving more reliable "unbanked" applicants.

## Solutions

Data Engineering Pipeline: An R-based pipeline that cleans anomalies (such as the 1000-year employment placeholder), handles missing values via training-set medians, and normalizes time offsets.

Advanced Feature Engineering: I engineered financial ratios—specifically Credit-to-Income, Annuity-to-Income, and Loan-to-Value (LTV)—to better capture borroIr stress.

Machine Learning Model: After evaluating Logistic Regression and Random Forest, I selected XGBoost. It outperformed simpler models by capturing non-linear interactions within the bureau and installment data, achieving a cross-validated AUC of 0.762.

## Business Value of the Solution

Precision in Risk Assessment: By moving beyond a simple "Majority Class" baseline (which was misleading due to the 8% default rate imbalance), My XGBoost model provides the discriminative poIr needed to flag defaults that traditional methods miss.

Revenue Growth: The model's ability to interpret alternative data allows the business to safely lend to customers who would otherwise be rejected under traditional banking conditions, expanding the customer base without increasing the relative risk.

Operational Efficiency: The automated aggregation of supplementary files (Bureau, Previous Applications, and Installments) reduces the manual effort required for underwriter review.

## My Contribution

I initially cleaned the data using R Studio (R). But then changed to Python once I moved my modeling notebook to Jupytr but still maintained the qualities of the cleaned data. Then, I created a model card to document model limitations and model use (also in Jupytr). 

## Difficulties Encountered

Data Leakage: I had to ensure that imputation (using medians) was calculated only on the training set and then applied to the test set to avoid biased results.

Class Imbalance: With only ~8% of applicants defaulting, standard accuracy was a poor metric. I had to pivot My strategy to focus on Area Under the Curve (AUC) to ensure the model actually learned the characteristics of a defaulter.

Computational Constraints: Processing large supplementary files (like Installment Payments) required efficient data aggregation to prevent memory errors during the join process.

## What I Learned

Importance of Domain Knowledge: Transforming "days" into "years" and creating debt-to-income ratios proved more predictive than raw data alone.

Model Transparency: Through building a Model Card (inspired by Google's Gemini 3.1 Flash Lite), I learned the importance of documenting model limitations and intended use cases for stakeholders.

Iterative Development: I learned how to refine code based on peer feedback (in the context of this project, since it's school related, the professor provided feedback on double checking the effectiveness of the model based on the AUC), specifically regarding the transition from the initial modeling phase to the final model card implementation.

## Repository Structure

data_preparation.R: The primary cleaning and feature engineering script.

Modeling_Assignment.ipynb: Contains the XGBoost training, cross-validation, and interpreted results.

Model_Card.pdf: Documentation of model performance and ethics.
