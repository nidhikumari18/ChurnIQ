"""Train and evaluate local churn classifiers on the IBM Telco dataset.

Run from the repository root: python model/train_model.py
Preprocessing is fitted only within each training-fold pipeline, after the
stratified train/test split. Evaluation outputs and the selected pipeline are
created locally under model/artifacts/ and are not committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import DEFAULT_DATASET_PATH, format_validation_report, load_dataset, validate_dataset

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create train-fitted imputing, scaling, and one-hot encoding transforms."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ], remainder="drop")


def _evaluate(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = pipeline.predict(x_test)
    classes = list(pipeline.named_steps["classifier"].classes_)
    yes_index = classes.index("Yes")
    positive_scores = pipeline.predict_proba(x_test)[:, yes_index]
    matrix = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_churn_yes": float(precision_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "recall_churn_yes": float(recall_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "f1_churn_yes": float(f1_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == "Yes").astype(int), positive_scores)),
        "confusion_matrix_labels": ["No", "Yes"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=["No", "Yes"], target_names=["No", "Yes"],
            output_dict=True, zero_division=0,
        ),
        "classification_report_text": classification_report(
            y_test, predictions, labels=["No", "Yes"], target_names=["No", "Yes"], zero_division=0,
        ),
    }


def train_and_evaluate(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    df = load_dataset(dataset_path)
    report = validate_dataset(df)
    print(format_validation_report(report))
    if not report["valid"]:
        raise ValueError("Training stopped because dataset validation failed. Resolve the reported errors first.")

    y = df[TARGET].astype(str).str.strip()
    x = df.drop(columns=[TARGET, *DROP_COLUMNS])
    numeric_features = [name for name in NUMERIC_FEATURES if name in x.columns]
    categorical_features = [name for name in x.columns if name not in numeric_features]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
    }
    artifact_dir = PROJECT_ROOT / "model" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results = []
    fitted: dict[str, Pipeline] = {}
    for name, factory in model_factories.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("classifier", factory()),
        ])
        pipeline.fit(x_train, y_train)
        metrics = _evaluate(name, pipeline, x_test, y_test)
        results.append(metrics)
        fitted[name] = pipeline
        display_metrics = {key: round(metrics[key], 4) for key in
                           ("accuracy", "precision_churn_yes", "recall_churn_yes", "f1_churn_yes", "roc_auc")}
        print(f"\n{name}: {display_metrics}\n{metrics['classification_report_text']}")

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"], display_labels=["No", "Yes"],
        ).plot(cmap="Purples", values_format="d", ax=ax, colorbar=False)
        ax.set(title=f"{name} — Churn Test Confusion Matrix", xlabel="Predicted churn", ylabel="Actual churn")
        fig.tight_layout()
        fig.savefig(artifact_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    # Select by positive-class F1, with churn recall as the tie-breaker; accuracy
    # alone is deliberately not the selection criterion.
    best = max(results, key=lambda row: (row["f1_churn_yes"], row["recall_churn_yes"]))
    best_pipeline = fitted[best["model"]]
    dump(best_pipeline, artifact_dir / "best_model.joblib")
    summary = {
        "dataset": "IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)",
        "target": TARGET,
        "positive_class": "Yes",
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "selection_rule": "Highest Churn=Yes F1-score; tie-break by Churn=Yes recall (not accuracy alone).",
        "best_model": best["model"],
        "metrics": results,
    }
    (artifact_dir / "evaluation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    score_rows = [{key: result[key] for key in
                   ("model", "accuracy", "precision_churn_yes", "recall_churn_yes", "f1_churn_yes", "roc_auc")}
                  for result in results]
    pd.DataFrame(score_rows).to_csv(artifact_dir / "model_comparison.csv", index=False)
    print(f"\nSelected by Churn=Yes F1 then recall: {best['model']}")
    print(f"Saved fitted pipeline and measured evaluation outputs to {artifact_dir}")
    return summary


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
