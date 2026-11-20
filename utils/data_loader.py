"""Load and validate the IBM Telco Customer Churn dataset.

The original CSV is never modified. TotalCharges is converted in memory;
blank values become NaN so the model pipeline can impute them safely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = 21
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

EXPECTED_COLUMNS_IN_DATASET = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}

EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    TARGET_COLUMN: {"Yes", "No"},
}


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the required IBM CSV, normalizing TotalCharges without editing it."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"IBM Telco dataset not found: {dataset_path}\n"
            "Download WA_Fn-UseC_-Telco-Customer-Churn.csv from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it at dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv. "
            "Do not substitute synthetic data."
        )
    try:
        frame = pd.read_csv(dataset_path, dtype={"TotalCharges": "string"})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read dataset CSV at {dataset_path}: {exc}") from exc

    if "TotalCharges" in frame.columns:
        # In this dataset blank strings represent missing values (not zero).
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
            errors="coerce",
        )
    return frame


def validate_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a structured validation report; errors do not hide other checks."""
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    rows, columns = frame.shape
    add("dimensions", "PASS" if columns == EXPECTED_COLUMNS else "FAIL",
        f"{rows} rows × {columns} columns; expected {EXPECTED_COLUMNS} original columns and about {EXPECTED_ROWS} rows.")
    actual = set(frame.columns)
    missing_columns = sorted(EXPECTED_COLUMNS_IN_DATASET - actual)
    extra_columns = sorted(actual - EXPECTED_COLUMNS_IN_DATASET)
    schema_ok = not missing_columns and not extra_columns
    schema_detail = "Column names match the IBM Telco schema."
    if missing_columns:
        schema_detail += f" Missing: {', '.join(missing_columns)}."
    if extra_columns:
        schema_detail += f" Unexpected: {', '.join(extra_columns)}."
    add("column schema", "PASS" if schema_ok else "FAIL", schema_detail)
    add("approximate row count", "PASS" if 7000 <= rows <= 7100 else "WARN",
        f"Found {rows} rows; the canonical dataset has approximately {EXPECTED_ROWS}.")

    if "TotalCharges" in frame.columns:
        nonblank = frame["TotalCharges"].notna().sum()
        blanks = int(frame["TotalCharges"].isna().sum())
        add("TotalCharges numeric conversion", "PASS" if pd.api.types.is_numeric_dtype(frame["TotalCharges"]) else "FAIL",
            f"Numeric values: {nonblank}; blank/invalid values represented as missing: {blanks}.")
    else:
        add("TotalCharges numeric conversion", "FAIL", "Column TotalCharges is missing.")

    missing_counts = frame.isna().sum()
    missing_detail = ", ".join(f"{col}={int(count)}" for col, count in missing_counts.items() if count)
    add("missing values", "INFO" if missing_counts.sum() else "PASS",
        missing_detail if missing_detail else "No missing values found.")
    duplicate_rows = int(frame.duplicated().sum())
    add("duplicate rows", "PASS" if duplicate_rows == 0 else "WARN", f"Found {duplicate_rows} duplicate row(s).")
    if ID_COLUMN in frame.columns:
        duplicate_ids = int(frame[ID_COLUMN].duplicated(keep=False).sum())
        blank_ids = int(frame[ID_COLUMN].isna().sum() + frame[ID_COLUMN].astype("string").str.strip().eq("").sum())
        add("customer IDs", "PASS" if duplicate_ids == 0 and blank_ids == 0 else "FAIL",
            f"Duplicate customer IDs (rows involved): {duplicate_ids}; blank IDs: {blank_ids}.")
    else:
        add("customer IDs", "FAIL", f"Required identifier column {ID_COLUMN} is missing.")

    if TARGET_COLUMN in frame.columns:
        observed_target = set(frame[TARGET_COLUMN].dropna().astype(str).str.strip().unique())
        unexpected = sorted(observed_target - EXPECTED_CATEGORIES[TARGET_COLUMN])
        target_ok = observed_target == EXPECTED_CATEGORIES[TARGET_COLUMN]
        detail = f"Observed target values: {', '.join(sorted(observed_target)) or 'none'}."
        if unexpected:
            detail += f" Unexpected values: {', '.join(unexpected)}."
        if frame[TARGET_COLUMN].isna().any():
            detail += f" Missing targets: {int(frame[TARGET_COLUMN].isna().sum())}."
        add("Churn target", "PASS" if target_ok and not frame[TARGET_COLUMN].isna().any() else "FAIL", detail)
    else:
        add("Churn target", "FAIL", f"Required target column {TARGET_COLUMN} is missing.")

    category_issues = []
    for column, allowed in EXPECTED_CATEGORIES.items():
        if column not in frame.columns:
            continue
        values = set(frame[column].dropna().astype(str).str.strip().unique())
        invalid = sorted(values - allowed)
        if invalid:
            category_issues.append(f"{column}: {', '.join(invalid)}")
    add("categorical values", "FAIL" if category_issues else "PASS",
        "; ".join(category_issues) if category_issues else "All observed categorical values match the IBM Telco schema.")

    errors = [item["detail"] for item in checks if item["status"] == "FAIL"]
    return {
        "valid": not errors,
        "shape": {"rows": rows, "columns": columns},
        "column_names": list(frame.columns),
        "data_types": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "checks": checks,
        "errors": errors,
    }


def format_validation_report(report: dict[str, Any]) -> str:
    """Format the structured report for a terminal or log."""
    lines = ["IBM Telco Customer Churn dataset validation", "=" * 45,
             f"Shape: {report['shape']['rows']} rows × {report['shape']['columns']} columns",
             "Columns: " + ", ".join(report["column_names"]), "Data types:"]
    lines.extend(f"  - {name}: {dtype}" for name, dtype in report["data_types"].items())
    lines.append("Checks:")
    lines.extend(f"  [{item['status']}] {item['check']}: {item['detail']}" for item in report["checks"])
    lines.append("Overall: " + ("VALID" if report["valid"] else "INVALID"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the IBM Telco churn CSV.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH,
                        help="Path to the IBM CSV (defaults to dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv).")
    args = parser.parse_args()
    try:
        frame = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report = validate_dataset(frame)
    print(format_validation_report(report))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
