"""Exploratory analysis for the official IBM Telco churn dataset.

Run from the repository root with: python notebooks/eda.py
Charts are written to static/images/eda/. No synthetic observations are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import DEFAULT_DATASET_PATH, format_validation_report, load_dataset, validate_dataset

OUTPUT_DIR = PROJECT_ROOT / "static" / "images" / "eda"


def _save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> None:
    df = load_dataset(dataset_path)
    report = validate_dataset(df)
    print(format_validation_report(report))
    if not report["valid"]:
        raise ValueError("EDA stopped because dataset validation failed. Fix the dataset issues above.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Purples")
    print("\nDataset overview\n", df.describe(include="all").transpose().to_string())
    print("\nDemographics (counts)\n", df[["gender", "SeniorCitizen", "Partner", "Dependents"]].describe().to_string())
    print("\nChurn distribution\n", df["Churn"].value_counts().to_string())
    for column in ["gender", "SeniorCitizen", "Contract", "InternetService", "PaymentMethod"]:
        print(f"\n{column} vs churn (within-group churn rate)\n",
              (pd.crosstab(df[column], df["Churn"], normalize="index") * 100).round(1).to_string())
    print("\nTenure by churn\n", df.groupby("Churn", observed=False)["tenure"].describe().round(2).to_string())
    for column in ["MonthlyCharges", "TotalCharges"]:
        print(f"\n{column} by churn\n", df.groupby("Churn", observed=False)[column].describe().round(2).to_string())
    service_columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                       "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    print("\nService adoption and churn rates (percent)\n",
          (df.groupby("Churn", observed=False)[service_columns].apply(
              lambda group: group.eq("Yes").mean() * 100)).round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df, x="Churn", hue="Churn", order=["No", "Yes"], legend=False, ax=ax)
    ax.set(title="Customer Churn Distribution", xlabel="Churn", ylabel="Number of customers")
    _save(fig, "churn_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn", order=["Month-to-month", "One year", "Two year"], ax=ax)
    ax.set(title="Customer Churn by Contract Type", xlabel="Contract type", ylabel="Number of customers")
    ax.legend(title="Churn", labels=["No", "Yes"])
    _save(fig, "contract_vs_churn.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="layer", element="step", stat="count", ax=ax)
    ax.set(title="Customer Tenure Distribution by Churn", xlabel="Tenure (months)", ylabel="Number of customers")
    ax.legend(title="Churn", labels=["No", "Yes"])
    _save(fig, "tenure_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, element="step", stat="count", common_norm=False, ax=ax)
    ax.set(title="Monthly Charges Distribution by Churn", xlabel="Monthly charges", ylabel="Number of customers")
    ax.legend(title="Churn", labels=["No", "Yes"])
    _save(fig, "monthly_charges_distribution.png")

    payment_rate = (pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100).reset_index().melt(
        id_vars="PaymentMethod", var_name="Churn", value_name="Customers (%)")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=payment_rate, x="PaymentMethod", y="Customers (%)", hue="Churn", ax=ax)
    ax.set(title="Churn Rate by Payment Method", xlabel="Payment method", ylabel="Customers within method (%)")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Churn")
    _save(fig, "payment_method_vs_churn.png")

    numeric = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
    correlation = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="Purples", vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"label": "Pearson correlation"}, ax=ax)
    ax.set(title="Correlation Heatmap: Numerical Features", xlabel="Numerical feature", ylabel="Numerical feature")
    _save(fig, "correlation_heatmap.png")
    print(f"\nSaved six charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        run_eda()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
