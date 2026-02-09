# 1. Clean and transform the application data based on your EDA findings by fixing: 

library(dplyr)
library(purrr)

#' Clean Application Data based on EDA
clean_application_data <- function(df) {
  df <- df %>%
    # Fix DAYS_EMPLOYED anomaly (365243 is a placeholder for 'Unemployed')
    mutate(DAYS_EMPLOYED_ANOM = as.integer(DAYS_EMPLOYED == 365243),
           DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED)) %>%
    # Fix other EDA issues: Convert negative day offsets to absolute positive values
    mutate(across(c(DAYS_BIRTH, DAYS_REGISTRATION, DAYS_ID_PUBLISH), abs))
    
  return(df)
}
  
  # 2. Create engineered features from the application train dataset

#' Create Engineered Features
engineer_features <- function(df) {
  df <- df %>%
    # Update demographics (Days to Years)
    mutate(AGE_YEARS = DAYS_BIRTH / 365.25,
           EMPLOYMENT_YEARS = abs(DAYS_EMPLOYED) / 365.25) %>%
    # Add Financial Ratios
    mutate(CREDIT_TO_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
           ANNUITY_TO_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL,
           LTV_RATIO = AMT_GOODS_PRICE / AMT_CREDIT) %>%
    # Add Missing Data Indicators (predictive of 'thin' credit files)
    mutate(EXT_SOURCE_1_NAN = as.integer(is.na(EXT_SOURCE_1)),
           EXT_SOURCE_2_NAN = as.integer(is.na(EXT_SOURCE_2)),
           EXT_SOURCE_3_NAN = as.integer(is.na(EXT_SOURCE_3))) %>%
    # Interaction term: Average of External Sources
    mutate(EXT_SOURCES_MEAN = rowMeans(select(., EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3), na.rm = TRUE))
    
  return(df)
}

# 3. Aggregate supplementary data to the applicant level (SK_ID_CURR)

#' Aggregate Relational Data
aggregate_supplementary <- function(bureau, prev, ins) {
  
  # Bureau: Prior credits and debt ratios
  b_agg <- bureau %>%
    group_by(SK_ID_CURR) %>%
    summarise(BUREAU_COUNT = n(),
              BUREAU_SUM_DEBT = sum(AMT_CRED_SUM_DEBT, na.rm = TRUE),
              BUREAU_DEBT_RATIO = sum(AMT_CRED_SUM_DEBT, na.rm = TRUE) / sum(AMT_CRED_SUM, na.rm = TRUE))

  # Previous Application: Approval and Refusal history
  prev_agg <- prev %>%
    group_by(SK_ID_CURR) %>%
    summarise(PREV_COUNT = n(),
              REFUSAL_COUNT = sum(NAME_CONTRACT_STATUS == "Refused"),
              APPROVAL_RATE = sum(NAME_CONTRACT_STATUS == "Approved") / n())

  # Installments: Late payment percentages
  ins_agg <- ins %>%
    mutate(IS_LATE = as.integer(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)) %>%
    group_by(SK_ID_CURR) %>%
    summarise(INSTAL_LATE_PCT = mean(IS_LATE, na.rm = TRUE))

  return(list(bureau = b_agg, prev = prev_agg, ins = ins_agg))
}

# 4. Join aggregated features to the application data

#' Join Aggregated Features to Application Data
join_features <- function(app_df, aggs) {
  df <- app_df %>%
    left_join(aggs$bureau, by = "SK_ID_CURR") %>%
    left_join(aggs$prev, by = "SK_ID_CURR") %>%
    left_join(aggs$ins, by = "SK_ID_CURR")
    
  return(df)
}

# 5. Ensure train/test consistency

#' Final Consistency and Alignment
ensure_consistency <- function(train_df, test_df) {
  # 1. Compute medians/bins on TRAINING data only
  train_ext_medians <- map_dbl(train_df[c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], median, na.rm = TRUE)
  age_bins <- quantile(train_df$AGE_YEARS, probs = seq(0, 1, 0.2), na.rm = TRUE)
  
  # 2. Apply to both sets
  apply_logic <- function(df) {
    # Impute using train medians
    df$EXT_SOURCE_1[is.na(df$EXT_SOURCE_1)] <- train_ext_medians["EXT_SOURCE_1"]
    df$EXT_SOURCE_2[is.na(df$EXT_SOURCE_2)] <- train_ext_medians["EXT_SOURCE_2"]
    df$EXT_SOURCE_3[is.na(df$EXT_SOURCE_3)] <- train_ext_medians["EXT_SOURCE_3"]
    # Apply train bins
    df$AGE_GROUP <- cut(df$AGE_YEARS, breaks = age_bins, include.lowest = TRUE, labels = FALSE)
    return(df)
  }
  
  train_df <- apply_logic(train_df)
  test_df  <- apply_logic(test_df)
  
  # 3. Produce identical columns (except TARGET)
  train_cols <- setdiff(names(train_df), "TARGET")
  test_df <- test_df[, intersect(names(test_df), train_cols), drop = FALSE]
  # Fill any missing columns in test with 0
  missing_cols <- setdiff(train_cols, names(test_df))
  test_df[missing_cols] <- 0
  test_df <- test_df[, train_cols] # Align order
  
  return(list(train = train_df, test = test_df))
}
