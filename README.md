# home-credit-project

This is my home credit risk demo project.

# In your .qmd file
from data_preparation import run_preparation_pipeline
# Now you can call the function directly

# Clean and transform the application data based on your EDA findings by fixing:

import pandas as pd
import numpy as np

def clean_application_data(df, state=None, is_train=True):
    """
    Cleans and transforms Home Credit application data based on EDA findings.
    
    Parameters:
    - df: The dataframe (train or test).
    - state: A dictionary to store/retrieve values like medians for consistency.
    - is_train: Boolean, True if processing training data.
    """
    df = df.copy()
    if state is None:
        state = {}

    # --- 1. Fix DAYS_EMPLOYED Anomaly ---
    # EDA shows 365243 is a placeholder for 'unemployed' or 'pensioner'.
    # We create a flag to keep that info, then replace the value with NaN.
    df['DAYS_EMPLOYED_ANOM'] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

    # --- 2. Handle EXT_SOURCE Missing Values ---
    # These are high-importance features. We fill missing values with the median.
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    if is_train:
        # Calculate and store medians from training data ONLY
        state['ext_medians'] = df[ext_cols].median()
    
    # Apply training medians to either train or test data
    df[ext_cols] = df[ext_cols].fillna(state['ext_medians'])

    # --- 3. Fix Other EDA Issues ---
    # DAYS_BIRTH is negative; convert to positive Age in years for readability
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    
    # Handle organic anomalies (e.g., negative DAYS_REGISTRATION, DAYS_ID_PUBLISH)
    cols_to_abs = ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
    for col in cols_to_abs:
        df[col] = abs(df[col])

    # Replace 0 values in AMT_INCOME_TOTAL if they exist (unlikely but safe)
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    if is_train:
        state['income_median'] = df['AMT_INCOME_TOTAL'].median()
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].fillna(state.get('income_median'))

    return df, state

# Example usage:
# train_df, training_state = clean_application_data(raw_train_df, is_train=True)
# test_df, _ = clean_application_data(raw_test_df, state=training_state, is_train=False)

# Create engineered features from the application train dataset

def engineer_application_features(df):
    """
    Requirement: Create engineered features from the application dataset.
    - Updates demographics (Age, Employment)
    - Adds financial ratios (Credit/Income, LTV, etc.)
    - Adds missing data indicators
    """
    df = df.copy()
    
    # --- 1. Demographic Updates ---
    # Convert negative days to positive years for interpretability
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365.25
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / -365.25
    
    # --- 2. Financial Ratios ---
    # Credit-to-Income: Total loan amount relative to annual income
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Annuity-to-Income: Monthly payment burden relative to income
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # Loan-to-Value (LTV): Ratio of the loan amount to the price of the goods
    # In this dataset, AMT_GOODS_PRICE / AMT_CREDIT represents the inverse of LTV
    df['GOODS_PRICE_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    
    # Income per Family Member: Captures available cash flow after basics
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    
    # Payment Rate: How much of the total credit is paid in one installment
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # --- 3. Missing Data Indicators ---
    # Indicators for EXT_SOURCE are often more predictive than the values themselves
    # because they may signal "thin" credit files or lack of history.
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[f'{col}_IS_MISSING'] = df[col].isnull().astype(int)
        
    # Indicator for OWN_CAR_AGE (missing usually means no car)
    df['OWN_CAR_AGE_IS_MISSING'] = df['OWN_CAR_AGE'].isnull().astype(int)

    return df

# Aggregate supplementary data to the applicant level (SK_ID_CURR)

def aggregate_bureau(bureau_df):
    """
    Requirement: Aggregate bureau.csv to applicant level.
    - Counts of prior credits
    - Active vs. Closed status
    - Total debt and overdue amounts
    - Debt-to-Credit ratios
    """
    # 1. Basic Aggregations
    b_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'AMT_CRED_SUM': 'sum',
        'AMT_CRED_SUM_DEBT': 'sum',
        'AMT_CRED_SUM_OVERDUE': 'sum',
        'DAYS_CREDIT': ['min', 'max', 'mean']
    })
    
    # Flatten multi-index columns
    b_agg.columns = ['BUREAU_' + '_'.join(col).strip() if isinstance(col, tuple) else 'BUREAU_' + col for col in b_agg.columns.values]
    
    # 2. Active vs Closed (Status Ratios)
    active_counts = bureau_df[bureau_df['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size()
    b_agg['BUREAU_ACTIVE_LOANS_COUNT'] = active_counts
    b_agg['BUREAU_ACTIVE_LOANS_COUNT'] = b_agg['BUREAU_ACTIVE_LOANS_COUNT'].fillna(0)
    
    # 3. Debt Ratio (Total Bureau Debt / Total Bureau Credit Limit)
    b_agg['BUREAU_DEBT_OVER_CREDIT'] = b_agg['BUREAU_AMT_CRED_SUM_DEBT_sum'] / b_agg['BUREAU_AMT_CRED_SUM_sum']
    
    return b_agg.fillna(0)

def aggregate_previous(prev_df):
    """
    Requirement: Aggregate previous

# Join aggregated features to the application data

def join_and_align_data(app_df, bureau_agg, prev_agg, ins_agg):
    """
    Requirement: Join aggregated features to the application data.
    - Uses SK_ID_CURR as the primary key.
    - Ensures all main application rows are preserved (Left Join).
    - Fills missing values in aggregated columns with 0.
    """
    # 1. Join Bureau data
    df = app_df.join(bureau_agg, on='SK_ID_CURR', how='left')
    
    # 2. Join Previous Applications data
    df = df.join(prev_agg, on='SK_ID_CURR', how='left')
    
    # 3. Join Installment Payments data
    df = df.join(ins_agg, on='SK_ID_CURR', how='left')
    
    # After joining, columns from supplementary tables will be NaN 
    # for applicants with no prior history. We fill these with 0.
    # Note: We only fill columns that were added during aggregation.
    agg_cols = (list(bureau_agg.columns) + 
                list(prev_agg.columns) + 
                list(ins_agg.columns))
    
    df[agg_cols] = df[agg_cols].fillna(0)
    
    return df

# Ensure train/test consistency

import pandas as pd
import numpy as np

def clean_application_data(df, state=None, is_train=True):
    """
    Cleans anomalies and transforms data based on EDA findings.
    
    Parameters:
    - df: DataFrame to process.
    - state: Dictionary storing values from training (medians, etc.).
    - is_train: Boolean to determine if we calculate or apply state.
    """
    df = df.copy()
    if state is None: state = {}

    # 1. Fix DAYS_EMPLOYED anomaly (365243 is a placeholder for 'Unemployed/Retired')
    # Create a flag to preserve the info, then set the value to NaN for imputation
    df['DAYS_EMPLOYED_ANOM'] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

    # 2. Impute EXT_SOURCE variables
    # These are highly predictive; we use training medians for consistency
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    if is_train:
        state['ext_medians'] = df[ext_cols].median()
    
    df[ext_cols] = df[ext_cols].fillna(state['ext_medians'])
    
    # 3. Handle negative time offsets
    # Convert 'Days Before' columns to absolute positive values
    time_cols = ['DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']
    for col in time_cols:
        df[col] = df[col].abs()
        
    return df, state

def engineer_features(df, state=None, is_train=True):
    """
    Requirement: Create engineered features including ratios and indicators.
    """
    df = df.copy()
    
    # 1. Update demographics (convert Days to Years)
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / 365.25
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'].abs() / 365.25
    
    # 2. Financial Ratios
    # Credit-to-Income: Overall debt burden
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # Annuity-to-Income: Monthly payment pressure
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # Loan-to-Value (LTV): Price of goods vs amount borrowed
    df['GOODS_PRICE_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT'] 
    
    # 3. Missing Data Indicators
    # Flagging missingness often captures signals about 'thin' credit history
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[f'{col}_IS_MISSING'] = df[col].isnull().astype(int)
    
    # 4. Interaction Term: Synergy of external sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    # 5. Binning Variables (Age)
    # Thresholds are calculated on Train and applied to Test
    if is_train:
        _, age_bins = pd.qcut(df['AGE_YEARS'], 5, retbins=True, labels=False)
        state['age_bins'] = age_bins
    
    df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], bins=state['age_bins'], 
                             labels=False, include_lowest=True)
    
    return df, state

def aggregate_supplementary(bureau_df, prev_df, ins_df):
    """
    Requirement: Aggregate Bureau, Previous Apps, and Installments.
    """
    # --- Bureau: Credit history summary ---
    b_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'AMT_CRED_SUM': 'sum',
        'AMT_CRED_SUM_DEBT': 'sum',
        'AMT_CRED_SUM_OVERDUE': 'sum'
    }).rename(columns={'SK_ID_BUREAU': 'BUREAU_COUNT'})
    
    # Active vs Closed
    active_mask = bureau_df['CREDIT_ACTIVE'] == 'Active'
    b_agg['BUREAU_ACTIVE_COUNT'] = bureau_df[active_mask].groupby('SK_ID_CURR').size()
    b_agg['BUREAU_DEBT_RATIO'] = b_agg['AMT_CRED_SUM_DEBT'] / b_agg['AMT_CRED_SUM'].replace(0, np.nan)

    # --- Previous Apps: Approval/Refusal patterns ---
    prev_agg = prev_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'NAME_CONTRACT_STATUS': lambda x: (x == 'Refused').sum()
    }).rename(columns={'SK_ID_PREV': 'PREV_APP_COUNT', 'NAME_CONTRACT_STATUS': 'PREV_REFUSAL_COUNT'})
    
    approved_mask = prev_df['NAME_CONTRACT_STATUS'] == 'Approved'
    prev_agg['PREV_APPROVAL_RATE'] = approved_mask.groupby(prev_df['SK_ID_CURR']).mean()

    # --- Installments: Repayment behavior ---
    ins_df['IS_LATE'] = (ins_df['DAYS_ENTRY_PAYMENT'] > ins_df['DAYS_INSTALMENT']).astype(int)
    ins_agg = ins_df.groupby('SK_ID_CURR')['IS_LATE'].agg(['mean', 'sum'])
    ins_agg.columns = ['INSTAL_LATE_PCT', 'INSTAL_LATE_COUNT']

    return b_agg.fillna(0), prev_agg.fillna(0), ins_agg.fillna(0)

def run_preparation_pipeline(app_df, bureau_df, prev_df, ins_df, state=None, is_train=True):
    """
    Master function to coordinate cleaning, engineering, aggregation, and joining.
    Ensures identical columns between Train and Test sets.
    """
    if state is None: state = {}

    # 1. Basic Cleaning & Transformation
    df, state = clean_application_data(app_df, state, is_train)
    
    # 2. Feature Engineering
    df, state = engineer_features(df, state, is_train)
    
    # 3. Aggregate Supplementary Files
    b_agg, p_agg, i_agg = aggregate_supplementary(bureau_df, prev_df, ins_df)
    
    # 4. Join all tables to Application Data
    df = df.join(b_agg, on='SK_ID_CURR', how='left')
    df = df.join(p_agg, on='SK_ID_CURR', how='left')
    df = df.join(i_agg, on='SK_ID_CURR', how='left')
    
    # 5. Handle missing values created by the join
    df = df.fillna(0)
    
    # 6. Ensure Column Consistency (Requirement: Identical columns except TARGET)
    if is_train:
        # Define the gold standard for columns based on training data
        state['final_columns'] = [c for c in df.columns if c != 'TARGET']
    else:
        # Align test data to the training columns, filling missing ones with 0
        target_col = ['TARGET'] if 'TARGET' in df.columns else []
        df = df.reindex(columns=state['final_columns'] + target_col, fill_value=0)
        
    return df, state
