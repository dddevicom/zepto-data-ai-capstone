# Module 2 — Analytics Pipeline

This module implements the 50-mark Zepto Analytics Pipeline as one cohesive workflow over the Seaborn Titanic dataset.

## Data loading

The raw dataset is loaded exactly once. If `titanic.csv` exists, it is read with `pandas.read_csv()`. Otherwise the script calls `sns.load_dataset("titanic")` once and immediately saves `titanic.csv`. All later EDA and modeling use the resulting DataFrame.

## Run

From the repository root:

```bash
cd analytics
python -m pip install -r requirements.txt
python analytics_pipeline.py
python reload_model.py
```

## Missing-value rule

The exact threshold rule is:

- `<5%` missing: drop affected rows
- `5%–30%`: impute
- `>30%`: drop the column

Exact measured percentages and strategies are written to `outputs/missing_strategy.csv`.

## EDA

The pipeline produces age/fare histograms and boxplots, IQR outlier counts, fare mean/median/mode and skewness interpretation, survival rates by sex/pclass/sex+pclass, the required six-column correlation matrix, the two strongest off-diagonal correlations, at least four multivariate charts with written interpretations, and before/after z-score checks for age and fare.

The correlation matrix uses exactly:

`survived, pclass, age, sibsp, parch, fare`

`adult_male` and `alone` are excluded from the correlation matrix.

## Modeling

A stratified train/test split is performed before preprocessing.

Classification preprocessing is fit on training data only and contains numeric imputation/scaling plus categorical imputation/one-hot encoding.

Three classifiers use the identical split:

- Logistic Regression
- Decision Tree
- Random Forest

The Decision Tree is rendered with `plot_tree` and labeled feature/class names.

All classifiers receive confusion matrix, accuracy, precision, recall, F1 and ROC/AUC evaluation.

## Imbalance

The same classifier is compared using:

1. baseline/no handling
2. `class_weight="balanced"`
3. SMOTE

SMOTE is inside an imbalanced-learn pipeline, so it is applied only to training data.

## Hyperparameter tuning

`GridSearchCV` tunes Random Forest:

- `n_estimators`
- `max_depth`
- `max_features`

The tuned estimator is constructed with `oob_score=True`, and the OOB score is reported.

## Regression

A multivariate Linear Regression predicts `fare` from the other available features. The module reports MAE, RMSE, R² and Adjusted R² and includes a residual plot with a written heteroscedasticity interpretation.

## Final artifact

The fitted preprocessing steps and tuned Random Forest are saved together as:

`outputs/best_complete_pipeline.joblib`

`reload_model.py` loads this complete pipeline and predicts directly from raw, unpreprocessed feature rows.

## Outputs

```text
analytics/
├── analytics_pipeline.py
├── reload_model.py
├── requirements.txt
├── README.md
├── titanic.csv
└── outputs/
    ├── charts/
    ├── classifier_metrics.csv
    ├── imbalance_comparison.csv
    ├── random_forest_tuning.json
    ├── regression_metrics.csv
    └── ...
```

All required written interpretations are saved as Markdown/text artifacts so the grader can assess the reasoning without relying on chart images alone.
