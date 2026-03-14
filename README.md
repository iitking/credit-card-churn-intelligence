# Credit Card Churn Intelligence

AI-Powered Customer Retention System.
<!-- milestone: 2 Add initial README with project overview and goals -->
<!-- milestone: 3 Add .gitignore for Python, OS, and virtual environments -->
<!-- milestone: 4 Add raw credit card customer dataset BankChurners.csv -->
<!-- milestone: 5 docs: document dataset column definitions and metadata -->
<!-- milestone: 6 feat: create basic requirements.txt draft -->
<!-- milestone: 7 eda: verify missing value counts across all 23 columns -->
<!-- milestone: 8 chore: setup project package configuration setup.py -->
<!-- milestone: 9 eda: add initial Jupyter notebook for exploratory analysis -->
<!-- milestone: 10 eda: inspect dataset shape and record counts in notebook -->
<!-- milestone: 11 eda: analyze target variable Attrition_Flag distribution -->
<!-- milestone: 12 eda: identify 16% class imbalance in churn rate -->
<!-- milestone: 13 eda: plot customer demographic age distribution -->
<!-- milestone: 14 eda: check gender distribution across existing vs attrited customers -->
<!-- milestone: 15 eda: plot boxplot of Avg_Open_To_Buy by churn status -->
<!-- milestone: 16 eda: examine education level distribution across cohorts -->
<!-- milestone: 17 eda: inspect marital status and dependent count correlation -->
<!-- milestone: 18 eda: analyze income category tiers and card categories -->
<!-- milestone: 19 eda: plot credit limit distribution and detect extreme right skew -->
<!-- milestone: 20 eda: analyze revolving balance behavior between churners and non-churners -->
<!-- milestone: 21 eda: analyze customer age distribution outliers -->
<!-- milestone: 22 eda: notice high concentration of zero revolving balance in churners -->
<!-- milestone: 23 eda: explore average utilization ratio metrics -->
<!-- milestone: 24 eda: investigate total transaction count and transaction amount -->
<!-- milestone: 25 eda: compute preliminary correlation coefficients with churn -->
<!-- milestone: 26 eda: find strong transaction count drop off in churned accounts -->
<!-- milestone: 27 eda: examine quarterly change metrics Q4 vs Q1 -->
<!-- milestone: 28 eda: investigate customer service contact count correlations -->
<!-- milestone: 29 eda: observe churn escalation with 3+ customer contacts in 12 months -->
<!-- milestone: 30 eda: plot months inactive distribution in past 12 months -->
<!-- milestone: 31 eda: generate automated profiling report output-2.html -->
<!-- milestone: 32 eda: document data summary and initial hypotheses -->
<!-- milestone: 33 docs: summarize key findings from exploratory data analysis -->
<!-- milestone: 34 refactor: create modular config directory and config.py -->
<!-- milestone: 35 config: define base paths and target column constants -->
<!-- milestone: 36 config: specify target mapping for Existing vs Attrited customer -->
<!-- milestone: 37 config: define Naive Bayes benchmark columns to drop -->
<!-- milestone: 38 config: define identifier column CLIENTNUM list -->
<!-- milestone: 39 feat: create src directory and src/__init__.py -->
<!-- milestone: 40 feat(data): implement initial data_loader.py skeleton -->
<!-- milestone: 41 feat(data): add safe CSV file loader with error handling -->
<!-- milestone: 42 feat(data): implement clean_data function to strip whitespace -->
<!-- milestone: 43 feat(data): add python logging to data loading steps -->
<!-- milestone: 44 feat(data): drop CLIENTNUM and benchmark classifier columns -->
<!-- milestone: 45 feat(data): add prepare_features_and_target function -->
<!-- milestone: 46 feat(data): implement stratified train test split with seed 42 -->
<!-- milestone: 47 feat(data): add input validation for missing target column -->
<!-- milestone: 48 test: create tests directory and tests/__init__.py -->
<!-- milestone: 49 test(data): add test_load_raw_data unit test -->
<!-- milestone: 50 test(data): add test_clean_data to verify dropped columns -->
<!-- milestone: 51 test(data): add test_prepare_features_and_target -->
<!-- milestone: 52 test(data): add test_get_train_test_split verifying stratification -->
<!-- milestone: 53 test(data): test data loader behavior on missing file path -->
<!-- milestone: 54 chore: setup pytest configuration and test runners -->
<!-- milestone: 55 notebook: update notebook with clean modular loading -->
<!-- milestone: 56 notebook: add correlation heatmap for numerical variables -->
<!-- milestone: 57 notebook: analyze relationship between credit limit and utilization -->
<!-- milestone: 58 notebook: explore multi-product customers vs single product churn -->
<!-- milestone: 59 notebook: plot transaction velocity against customer tenure -->
<!-- milestone: 60 notebook: add pairwise scatterplots for top numerical features -->
<!-- milestone: 61 notebook: check distribution of inactive months in last year -->
<!-- milestone: 62 notebook: inspect customer contact count friction signals -->
<!-- milestone: 63 notebook: compare churn proportion across card categories -->
<!-- milestone: 64 notebook: check income tier differences between genders -->
<!-- milestone: 65 refactor(data): enforce strict column ordering in clean_data -->
<!-- milestone: 66 refactor(data): optimize data_loader memory usage and type casting -->
<!-- milestone: 67 feat(features): create src/features.py module -->
<!-- milestone: 68 config: define ordinal hierarchical column orders in config.py -->
<!-- milestone: 69 config: set Income_Category ordinal levels from Unknown to $120K+ -->
<!-- milestone: 70 config: set Card_Category levels Blue, Silver, Gold, Platinum -->
<!-- milestone: 71 config: set Education_Level hierarchy from Uneducated to Doctorate -->
<!-- milestone: 72 config: define nominal columns Gender and Marital_Status -->
<!-- milestone: 73 config: define base continuous numerical column names -->
<!-- milestone: 74 feat(features): implement FeatureEngineer BaseEstimator class -->
<!-- milestone: 75 feat(features): add Avg_Transaction_Amt derivation -->
<!-- milestone: 76 feat(features): add Trans_Count_Per_Month velocity feature -->
<!-- milestone: 77 feat(features): add unit test for zero transaction count edge case -->
<!-- milestone: 78 feat(features): add Trans_Amt_Per_Month spending rate -->
<!-- milestone: 79 feat(features): compute Inactive_Ratio relative to 12 months -->
<!-- milestone: 80 feat(features): calculate Contact_Intensity support ratio -->
<!-- milestone: 81 feat(features): calculate Revolving_to_Credit_Ratio -->
