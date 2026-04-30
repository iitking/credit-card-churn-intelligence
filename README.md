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
<!-- milestone: 82 feat(features): compute Total_Products_Per_Year relationship deepening -->
<!-- milestone: 83 feat(features): verify numerical feature scaling bounds -->
<!-- milestone: 84 feat(features): add zero-division safety handling in FeatureEngineer -->
<!-- milestone: 85 feat(features): implement build_preprocessor with ColumnTransformer -->
<!-- milestone: 86 feat(features): configure OrdinalEncoder for hierarchy columns -->
<!-- milestone: 87 feat(features): configure OneHotEncoder with drop='first' -->
<!-- milestone: 88 feat(features): add StandardScaler for all numerical features -->
<!-- milestone: 89 feat(features): implement build_full_feature_pipeline Pipeline -->
<!-- milestone: 90 feat(features): add unknown value handling to OrdinalEncoder -->
<!-- milestone: 91 feat(features): implement get_feature_names extraction utility -->
<!-- milestone: 92 test(features): create test_features.py with unit tests -->
<!-- milestone: 93 test(features): test derived ratios mathematical precision -->
<!-- milestone: 94 test(features): test ColumnTransformer fit_transform shapes -->
<!-- milestone: 95 test(features): test feature names mapping accuracy -->
<!-- milestone: 96 test(features): test feature names alignment with transformed matrix -->
<!-- milestone: 97 notebook: integrate FeatureEngineer into notebook experiments -->
<!-- milestone: 98 notebook: verify transformed feature matrix shape (8101, 28) -->
<!-- milestone: 99 refactor(features): add type annotations to transform pipeline -->
<!-- milestone: 100 docs: document feature engineering rationale and domain formulas -->
<!-- milestone: 101 docs: add mathematical formulas for engineered banking ratios -->
<!-- milestone: 102 chore: update requirements with scikit-learn and scipy versions -->
<!-- milestone: 103 feat(models): create src/models.py candidate model factory -->
<!-- milestone: 104 feat(models): add LogisticRegression baseline with class_weight='balanced' -->
<!-- milestone: 105 feat(models): add RandomForestClassifier with balanced_subsample -->
<!-- milestone: 106 feat(models): add GradientBoostingClassifier configuration -->
<!-- milestone: 107 feat(models): add ExtraTreesClassifier candidate model -->
<!-- milestone: 108 feat(models): add random_state consistency across all models -->
<!-- milestone: 109 feat(models): add HistGradientBoostingClassifier -->
<!-- milestone: 110 feat(models): integrate XGBoost with scale_pos_weight for class imbalance -->
<!-- milestone: 111 feat(models): integrate LightGBM with scale_pos_weight -->
<!-- milestone: 112 feat(models): add resilient fallback when native libraries missing -->
<!-- milestone: 113 feat(models): add create_ensemble_model soft voting utility -->
<!-- milestone: 114 feat(models): implement soft voting ensemble architecture -->
<!-- milestone: 115 feat(eval): create src/evaluate.py module -->
<!-- milestone: 116 feat(eval): implement calculate_metrics for Accuracy, F1, ROC, PR-AUC -->
<!-- milestone: 117 feat(eval): calculate confusion matrix TP, FP, TN, FN breakdown -->
<!-- milestone: 118 feat(eval): implement banking business impact metrics in evaluate.py -->
<!-- milestone: 119 feat(eval): implement find_optimal_threshold search routine -->
<!-- milestone: 120 feat(eval): add print_evaluation_summary formatted report -->
<!-- milestone: 121 feat(eval): add PR-AUC (Average Precision) metric calculation -->
<!-- milestone: 122 feat(train): create src/train.py pipeline script -->
<!-- milestone: 123 feat(train): implement 5-fold StratifiedKFold cross validation -->
<!-- milestone: 124 feat(train): compute cross validation F1 and ROC-AUC means -->
<!-- milestone: 125 feat(train): benchmark candidate models and rank by test F1 -->
<!-- milestone: 126 feat(train): record standard deviation of cross-validation folds -->
<!-- milestone: 127 feat(train): select champion model (XGBoost) based on benchmarks -->
<!-- milestone: 128 feat(train): calibrate champion model threshold for max ROI -->
<!-- milestone: 129 feat(train): assemble full end-to-end Pipeline with preprocessing -->
<!-- milestone: 130 feat(train): export best_churn_pipeline.joblib and pickle model -->
<!-- milestone: 131 feat(train): save model_metadata.json with metrics and parameters -->
<!-- milestone: 132 feat(train): save metrics_summary.json benchmark table -->
<!-- milestone: 133 feat(train): verify model persistence and file sizes -->
<!-- milestone: 134 notebook: run Optuna hyperparameter study in notebook -->
<!-- milestone: 135 notebook: test best hyperparameter settings on validation fold -->
<!-- milestone: 136 notebook: evaluate final classifier on holdout set -->
<!-- milestone: 137 docs: update model benchmark results table in README -->
