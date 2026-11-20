# ChurnIQ

ChurnIQ is being built in small, testable phases as a local-first customer churn analysis application. It will use the IBM Telco Customer Churn dataset as its sole customer dataset and will not invent customer records, model metrics, or prediction probabilities.

## Phase 1: Flask foundation and landing page

### Requirements
- Python 3.9 or newer
- Internet only to install Flask the first time (the application itself runs locally)

### Run locally

From the repository folder:

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install Flask and start the app:

```bash
python -m pip install -r requirements.txt
flask --app app run --debug
```

Open <http://127.0.0.1:5000> in a browser. The health endpoint is at <http://127.0.0.1:5000/health>.

### Phase 1 checks

1. Confirm the landing page loads and adapts to a narrow browser window.
2. Visit `/health`; it should return JSON with `status` set to `ok`.
3. Confirm no customer records, prediction scores, or model performance values are shown; the IBM dataset and real trained models have not been added yet.

## Planned phases

- Phase 2: inspect/add and validate the IBM Telco dataset; build local preprocessing and model-training/evaluation pipeline.
- Phase 3: customer and prediction views backed by the real dataset and trained pipeline.
- Phase 4: SQLite authentication, prediction history, and security hardening.
- Phase 5: optional explanation integration, with an explicit rule-based fallback that works without an API key.

Each phase will be validated before moving to the next. The optional AI integration will not be required for the application to work.
