
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    roc_curve
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
CHARTS = OUTPUTS / "charts"
CSV_PATH = BASE / "titanic.csv"
MODEL_PATH = OUTPUTS / "best_complete_pipeline.joblib"

RANDOM_STATE = 42
TARGET = "survived"
CLASS_FEATURES = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
CORR_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]


def setup():
    OUTPUTS.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)


def load_dataset_once():
    # Required design: exactly one raw-data load.
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    df = sns.load_dataset("titanic")
    df.to_csv(CSV_PATH, index=False)
    return df


def save_text(filename, text):
    (OUTPUTS / filename).write_text(text, encoding="utf-8")


def profile(df):
    buf = io.StringIO()
    df.info(buf=buf)
    save_text("profile_info.txt", buf.getvalue())
    df.describe(include="all").to_csv(OUTPUTS / "describe.csv")
    pd.DataFrame({"rows": [df.shape[0]], "columns": [df.shape[1]]}).to_csv(
        OUTPUTS / "shape.csv", index=False
    )
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing[missing > 0].to_csv(
        OUTPUTS / "missing_percentages.csv", header=["missing_pct"]
    )
    return missing[missing > 0]


def clean(df):
    work = df.copy()
    missing = work.isna().mean() * 100
    decisions = []

    for col, pct in missing[missing > 0].items():
        if pct < 5:
            work = work.dropna(subset=[col])
            strategy = "drop affected rows"
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(work[col]):
                work[col] = work[col].fillna(work[col].median())
                strategy = "median imputation"
            else:
                work[col] = work[col].fillna(work[col].mode(dropna=True).iloc[0])
                strategy = "mode imputation"
        else:
            work = work.drop(columns=[col])
            strategy = "drop column"

        decisions.append({
            "column": col,
            "missing_pct": round(float(pct), 4),
            "strategy": strategy
        })

    pd.DataFrame(decisions).to_csv(
        OUTPUTS / "missing_strategy.csv", index=False
    )
    return work


def save_chart(fig, filename):
    fig.tight_layout()
    fig.savefig(CHARTS / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def iqr_count(s):
    q1 = s.quantile(.25)
    q3 = s.quantile(.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return q1, q3, iqr, lo, hi, int(((s < lo) | (s > hi)).sum())


def eda(df):
    rows = []
    for col in ["age", "fare"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=25)
        ax.set_title(f"{col.title()} Histogram")
        save_chart(fig, f"{col}_histogram.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(df[col].dropna(), vert=False)
        ax.set_title(f"{col.title()} Box Plot")
        save_chart(fig, f"{col}_boxplot.png")

        rows.append([col, *iqr_count(df[col])])

    pd.DataFrame(
        rows,
        columns=["column", "Q1", "Q3", "IQR", "lower_fence",
                 "upper_fence", "outlier_count"]
    ).to_csv(OUTPUTS / "iqr_outlier_counts.csv", index=False)

    mean = df["fare"].mean()
    median = df["fare"].median()
    mode = df["fare"].mode().iloc[0]
    if mean > median > mode:
        conclusion = "Fare is right-skewed because mean > median > mode."
    elif mean < median < mode:
        conclusion = "Fare is left-skewed because mean < median < mode."
    else:
        conclusion = "Fare does not follow a simple mean/median/mode skew ordering."

    save_text(
        "fare_skewness_interpretation.txt",
        f"Mean: {mean:.4f}\nMedian: {median:.4f}\nMode: {mode:.4f}\n"
        f"Conclusion: {conclusion}\n"
    )

    sex = df.groupby("sex", observed=True)[TARGET].mean().mul(100).round(2)
    pclass = df.groupby("pclass", observed=True)[TARGET].mean().mul(100).round(2)
    sex_pclass = (
        df.groupby(["sex", "pclass"], observed=True)[TARGET]
        .mean().mul(100).round(2)
    )
    sex.to_csv(OUTPUTS / "survival_rate_by_sex.csv")
    pclass.to_csv(OUTPUTS / "survival_rate_by_pclass.csv")
    sex_pclass.to_csv(OUTPUTS / "survival_rate_by_sex_pclass.csv")

    corr = df[CORR_COLUMNS].corr()
    corr.to_csv(OUTPUTS / "correlation_matrix.csv")

    pairs = []
    for i, a in enumerate(CORR_COLUMNS):
        for j in range(i + 1, len(CORR_COLUMNS)):
            b = CORR_COLUMNS[j]
            value = corr.loc[a, b]
            pairs.append((a, b, value, abs(value)))

    top2 = pd.DataFrame(
        sorted(pairs, key=lambda x: x[3], reverse=True)[:2],
        columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
    )
    top2.to_csv(OUTPUTS / "top_two_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Matrix — Required Six Columns")
    save_chart(fig, "correlation_heatmap.png")

    return top2


def multivariate_story(df):
    interpretations = [
        "Passenger class and sex jointly show a strong survival pattern, with survival differing across classes and generally higher for females.",
        "Fare varies substantially by passenger class, and survival differences can be compared within each class.",
        "Age distributions overlap between survivors and non-survivors, but age still adds information to the passenger profile.",
        "The age-fare scatter shows interaction among fare, age, class and survival rather than a single-variable story.",
        "Survival composition differs by embarkation port, although passenger mix and class may partly explain the relationship."
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    rates = df.groupby(["pclass", "sex"], observed=True)["survived"].mean().unstack()
    rates.plot(kind="bar", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival Rate by Passenger Class and Sex")
    save_chart(fig, "01_survival_by_pclass_sex.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = []
    labels = []
    for pclass in sorted(df["pclass"].unique()):
        for survived in [0, 1]:
            groups.append(df.loc[(df["pclass"] == pclass) & (df["survived"] == survived), "fare"].dropna())
            labels.append(f"P{pclass}-S{survived}")
    ax.boxplot(groups, labels=labels)
    ax.set_ylabel("Fare")
    ax.set_title("Fare by Class and Survival")
    save_chart(fig, "02_fare_by_class_survival.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for survived in [0, 1]:
        ax.hist(df.loc[df["survived"] == survived, "age"].dropna(), bins=25, alpha=.5, label=f"Survived={survived}")
    ax.set_xlabel("Age")
    ax.set_title("Age Distribution by Survival")
    ax.legend()
    save_chart(fig, "03_age_by_survival.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for survived in [0, 1]:
        subset = df[df["survived"] == survived]
        ax.scatter(subset["fare"], subset["age"], alpha=.6, label=f"Survived={survived}")
    ax.set_xlabel("Fare")
    ax.set_ylabel("Age")
    ax.set_title("Age vs Fare by Survival")
    ax.legend()
    save_chart(fig, "04_age_fare_survival_class.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df.groupby(["embarked", "survived"], observed=True).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Passenger count")
    ax.set_title("Survival Counts by Embarkation Port")
    save_chart(fig, "05_survival_by_embarked.png")

    save_text(
        "multivariate_chart_interpretations.md",
        "\n".join(f"- {x}" for x in interpretations)
    )


def standardization_check(df):
    rows = []
    for col in ["age", "fare"]:
        x = df[col].astype(float)
        z = (x - x.mean()) / x.std(ddof=1)
        rows.append({
            "column": col,
            "before_mean": x.mean(),
            "before_std": x.std(ddof=1),
            "after_mean": z.mean(),
            "after_std": z.std(ddof=1)
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(x, bins=25, alpha=.5, label="Before", density=True)
        ax.hist(z, bins=25, alpha=.5, label="After z-score", density=True)
        ax.set_title(f"Before/After Standardization: {col}")
        ax.legend()
        save_chart(fig, f"{col}_standardization_check.png")

    pd.DataFrame(rows).to_csv(
        OUTPUTS / "standardization_check.csv", index=False
    )


def preprocessor():
    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]

    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", num, numeric),
        ("cat", cat, categorical)
    ])


def clf_pipe(model):
    return Pipeline([
        ("preprocessor", preprocessor()),
        ("model", model)
    ])


def evaluate(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "pred": pred,
        "proba": proba
    }


def classification(df):
    X = df[CLASS_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=RANDOM_STATE
    )

    pd.DataFrame({
        "split": ["train", "test"],
        "rows": [len(y_train), len(y_test)],
        "survived_rate": [y_train.mean(), y_test.mean()]
    }).to_csv(OUTPUTS / "stratified_split_class_balance.csv", index=False)

    models = {
        "Logistic Regression": clf_pipe(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": clf_pipe(
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
        ),
        "Random Forest": clf_pipe(
            RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, oob_score=True
            )
        )
    }

    results = []
    roc_lines = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        ev = evaluate(name, model, X_test, y_test)
        results.append({
            k: v for k, v in ev.items()
            if k not in ["pred", "proba"]
        })
        fpr, tpr, _ = roc_curve(y_test, ev["proba"])
        roc_lines.append((name, fpr, tpr, ev["auc"]))

    metrics = pd.DataFrame(results)
    metrics.to_csv(OUTPUTS / "classifier_metrics.csv", index=False)

    with (OUTPUTS / "confusion_matrices.txt").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['model']}\n{np.array(r['confusion_matrix'])}\n\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, fpr, tpr, auc in roc_lines:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Three Classifiers")
    ax.legend()
    save_chart(fig, "classifier_roc_comparison.png")

    tree_model = models["Decision Tree"]
    feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()
    tree = tree_model.named_steps["model"]
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        tree, feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        filled=True, rounded=True, max_depth=4, fontsize=8, ax=ax
    )
    ax.set_title("Decision Tree — Labeled Features and Classes")
    save_chart(fig, "decision_tree.png")

    imbalance = {
        "baseline_no_handling": clf_pipe(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "class_weight_balanced": clf_pipe(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)
        ),
        "smote_training_only": ImbPipeline([
            ("preprocessor", preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ])
    }

    imbalance_rows = []
    for name, model in imbalance.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        imbalance_rows.append({
            "variant": name,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0)
        })

    imb_df = pd.DataFrame(imbalance_rows)
    imb_df.to_csv(OUTPUTS / "imbalance_comparison.csv", index=False)
    best_imb = imb_df.sort_values("f1", ascending=False).iloc[0]
    save_text(
        "imbalance_conclusion.txt",
        f"Best imbalance variant by F1: {best_imb['variant']} "
        f"(F1={best_imb['f1']:.4f}). SMOTE is inside a training pipeline, "
        "so oversampling is performed only on training data."
    )

    rf = clf_pipe(
        RandomForestClassifier(
            random_state=RANDOM_STATE, oob_score=True, n_jobs=-1
        )
    )
    grid = GridSearchCV(
        rf,
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"]
        },
        scoring="roc_auc",
        cv=5, n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_ev = evaluate("Tuned Random Forest", best, X_test, y_test)

    tuning = {
        "best_params": grid.best_params_,
        "best_cv_auc": float(grid.best_score_),
        "test_auc": float(best_ev["auc"]),
        "oob_score": float(best.named_steps["model"].oob_score_)
    }
    save_text("random_forest_tuning.json", json.dumps(tuning, indent=2))

    joblib.dump(best, MODEL_PATH)
    return metrics, best_ev, X_train, X_test, y_train, y_test


def regression(df):
    features = ["pclass", "age", "sibsp", "parch", "sex", "embarked", "survived"]
    X = df[features]
    y = df["fare"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE
    )

    num = ["pclass", "age", "sibsp", "parch", "survived"]
    cat = ["sex", "embarked"]
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat)
    ])

    model = Pipeline([
        ("preprocessor", prep),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    n = len(y_test)
    p = model.named_steps["preprocessor"].transform(X_test).shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    metrics = {
        "MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adjusted_r2
    }
    pd.DataFrame([metrics]).to_csv(
        OUTPUTS / "regression_metrics.csv", index=False
    )

    residuals = y_test.to_numpy() - pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, residuals, alpha=.65)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted fare")
    ax.set_ylabel("Residual")
    ax.set_title("Linear Regression Residual Plot")
    save_chart(fig, "regression_residual_plot.png")

    corr = abs(np.corrcoef(pred, residuals)[0, 1])
    if corr > .15:
        text = "The residuals show a non-random spread pattern, suggesting possible heteroscedasticity."
    else:
        text = "The residual plot does not show strong evidence of heteroscedasticity from this visual/correlation check."
    save_text("regression_residual_interpretation.txt", text)

    return metrics


def final_comparison(class_metrics, tuned, reg_metrics):
    table = class_metrics[["model", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    table = pd.concat([
        table,
        pd.DataFrame([{
            "model": "Tuned Random Forest",
            "accuracy": tuned["accuracy"],
            "precision": tuned["precision"],
            "recall": tuned["recall"],
            "f1": tuned["f1"],
            "auc": tuned["auc"]
        }])
    ], ignore_index=True)

    table.to_csv(OUTPUTS / "classification_model_comparison.csv", index=False)
    pd.DataFrame([reg_metrics]).to_csv(
        OUTPUTS / "regression_model_comparison.csv", index=False
    )

    best = table.sort_values(["f1", "auc"], ascending=False).iloc[0]
    save_text(
        "final_recommendation.md",
        f"## Final Recommendation\n\n"
        f"Recommended classifier: **{best['model']}**. "
        f"It has the strongest combined F1/AUC in the reported comparison "
        f"(F1={best['f1']:.3f}, AUC={best['auc']:.3f}).\n\n"
        "Classification and regression metrics are kept as separate metric groups "
        "because they measure different tasks and are not directly comparable. "
        "For deployment, the complete saved classification pipeline is preferred "
        "because preprocessing and the final estimator are stored together."
    )


def main():
    setup()

    print("1/8 Loading Titanic dataset once...")
    raw = load_dataset_once()
    print("Raw shape:", raw.shape)

    print("2/8 Profiling...")
    profile(raw)

    print("3/8 Cleaning...")
    df = clean(raw)
    df.to_csv(OUTPUTS / "cleaned_titanic.csv", index=False)
    print("Cleaned shape:", df.shape)

    print("4/8 EDA and data story...")
    top2 = eda(df)
    multivariate_story(df)
    standardization_check(df)
    print("Top two correlations:")
    print(top2)

    print("5/8 Classification...")
    class_metrics, tuned, *_ = classification(df)

    print("6/8 Regression...")
    reg_metrics = regression(df)

    print("7/8 Final comparison...")
    final_comparison(class_metrics, tuned, reg_metrics)

    print("8/8 Complete pipeline saved.")
    print("Model:", MODEL_PATH)

    reload_code = '''from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
pipeline = joblib.load(BASE / "outputs" / "best_complete_pipeline.joblib")

raw = pd.DataFrame([
    {"pclass": 3, "age": 30, "sibsp": 0, "parch": 0,
     "fare": 8.05, "sex": "male", "embarked": "S"},
    {"pclass": 1, "age": 35, "sibsp": 1, "parch": 0,
     "fare": 100.0, "sex": "female", "embarked": "C"}
])

print("Reloaded:", type(pipeline).__name__)
print("Raw-input predictions:", pipeline.predict(raw).tolist())
'''
    (BASE / "reload_model.py").write_text(
        reload_code, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
