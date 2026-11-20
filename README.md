# ChurnIQ

ChurnIQ is being built in small, testable phases as a local-first customer churn analysis application. It uses only the IBM Telco Customer Churn dataset for customer data; it does not fabricate customer records, prediction scores, or model metrics.

## Phase 1: Flask foundation and landing page

### Requirements
- Python 3.9 or newer
- Internet only to install packages (the application itself runs locally)

### Run locally

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages and start the app:

```bash
python -m pip install -r requirements.txt
flask --app app run --debug
```

Open <http://127.0.0.1:5000>; health check: <http://127.0.0.1:5000/health>.

## Phase 2: dataset validation, EDA, and model training

### 1. Download the required dataset

Download the original `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the [IBM Telco Customer Churn Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Place the CSV at:

```text
dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is not bundled because of its third-party distribution terms. If it is absent, the validation and analysis commands stop with download instructions. Do not generate substitute or synthetic customer data.

### 2. Install and validate

```bash
python -m pip install -r requirements.txt
python utils/data_loader.py
```

The loader checks the IBM 21-column schema, approximate row count, target values, category values, duplicate rows and IDs, missing values, dtypes, and numeric `TotalCharges`. Blank TotalCharges are missing values and remain missing for pipeline imputation; the source CSV is never changed.

### 3. Generate exploratory charts

```bash
python notebooks/eda.py
```

This prints summaries for demographics, churn groups, contract, internet service, payment method, tenure, charges, and services. It saves six dataset-derived charts in `static/images/eda/`.

### 4. Train and compare models

```bash
python model/train_model.py
```

The script makes a stratified 80/20 split (random state 42), then fits imputation, scaling, and categorical encoding inside each model pipeline to avoid test-data leakage. It compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using accuracy, Churn=Yes precision/recall/F1, ROC-AUC, confusion matrices, and classification reports. Selection uses Churn=Yes F1 and recall—not accuracy alone.

Generated local outputs (not committed) are in `model/artifacts/`: `best_model.joblib`, `evaluation_results.json`, `model_comparison.csv`, and four confusion-matrix charts. Values are calculated from the downloaded dataset and current training run; no example metrics are hardcoded.

### Phase 2 checks

1. Run `python utils/data_loader.py`; confirm a valid report and 7,043 × 21 canonical shape.
2. Run `python notebooks/eda.py`; confirm six charts exist under `static/images/eda/`.
3. Run `python model/train_model.py`; inspect Churn=Yes recall, F1, ROC-AUC and confusion matrices for all four models.
4. If the dataset is missing, verify the commands give the Kaggle source and expected path rather than inventing data.

## Planned phases

- Phase 3: customer and prediction views connected to the genuine trained pipeline.
- Phase 4: SQLite authentication, prediction history, and security hardening.
- Phase 5: optional explanation integration, with an explicit rule-based fallback that works without an API key.

The optional AI integration will not be required for the application to work.
