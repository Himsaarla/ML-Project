Loan Approval Prediction Using Ensemble Learning
A full-stack academic project that predicts loan approval outcomes using Random Forest, XGBoost, LightGBM, and a soft-voting ensemble, with dataset exploration, model evaluation, and SHAP explainability.
> **Academic-use notice:** This project is for educational purposes only. Its predictions are not actual bank decisions and must not be used as the sole basis for real lending decisions.
Project team
Khyati Karri — `2520030279`
Prashanthi Dabbugunta — `2520030301`
Aarla Himesh — `2520030317`
Project guide: Dr. Y. Lakshmi Prasanna
Features
Dataset explorer with search, filtering, sorting, pagination, and CSV download
Dataset summary and exploratory data analysis (EDA)
Applicant form for individual predictions
Random Forest, XGBoost, LightGBM, and soft-voting ensemble
Evaluation using accuracy, precision, recall, F1-score, F2-score, ROC-AUC, and confusion matrices
Dataset prediction output with actual and predicted statuses shown separately
SHAP-based model explanations where supported
Downloadable prediction results
Input validation and API error handling
Only show metrics and predictions computed from the actual dataset and trained models. Do not use placeholder values or fabricated results.
Dataset
Place the supplied CSV at:
```text
backend/data/loan_approval_dataset.csv
```
The dataset is expected to include these columns:
```text
loan_id
no_of_dependents
education
self_employed
income_annum
loan_amount
loan_term
cibil_score
residential_assets_value
commercial_assets_value
luxury_assets_value
bank_asset_value
loan_status
```
The CSV may contain extra whitespace in column names and categorical values. The data-loading pipeline should normalize whitespace while preserving the original file.
`loan_id` is an identifier and must not be used as a model feature.
`loan_status` is the target and must never be passed into the model as an input feature.
For risk-focused evaluation, use Rejected = 1 and Approved = 0, and document the encoding.
Display original labels separately from predictions.
Report actual missing-value and duplicate counts; do not silently fabricate, duplicate, or remove records.
Engineered features
If implemented, document and safely calculate the following:
`loan_to_income = loan_amount / income_annum`
`loan_to_assets = loan_amount / total_asset_value`
`payment_to_income = loan_amount / (loan_term * income_annum)`
Protect calculations against zero denominators and invalid values. Ensure feature engineering is consistent between training and single-applicant predictions.
Technology stack
Frontend
React
Vite
JavaScript
Recharts
Lucide React
CSS
Backend
Python
Flask
pandas and NumPy
scikit-learn
XGBoost
LightGBM
SHAP
joblib
The frontend communicates with the Flask backend through a REST API.
Project structure
```text
loan-approval-prediction/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── styles/
│   └── package.json
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── data/
│   │   └── loan_approval_dataset.csv
│   ├── models/
│   ├── services/
│   │   ├── data_service.py
│   │   ├── training_service.py
│   │   ├── prediction_service.py
│   │   ├── evaluation_service.py
│   │   └── explainability_service.py
│   └── tests/
├── .gitignore
└── README.md
```
Your implementation may use a different structure if the responsibilities remain clear.
Setup and run
These commands assume the project has the folder structure above and that Python and Node.js are installed.
1. Add the dataset
Copy `loan_approval_dataset.csv` into `backend/data/`.
2. Start the backend
Windows PowerShell:
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
If PowerShell blocks virtual-environment activation, use Command Prompt and run:
```bat
.venv\Scripts\activate.bat
```
If the `py` command is unavailable, use `python -m venv .venv`.
The backend should print the local address on which Flask is running. Keep this terminal open.
3. Start the frontend
Open a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open the local URL printed by Vite, usually `http://localhost:5173`.
> The exact commands and URLs may differ depending on the implementation. Follow the scripts in `package.json` and the settings in the backend configuration.
Model training and evaluation methodology
The intended workflow is:
Load and validate the supplied dataset.
Normalize whitespace in column names and categorical values.
Separate the target from the model features and exclude `loan_id`.
Encode categorical features consistently.
Split the data into training and test sets using an 80/20 stratified split and `random_state=42`.
Fit preprocessing and models on the training data only.
Train Random Forest, XGBoost, and LightGBM.
Combine the models with soft voting by averaging predicted class probabilities.
Evaluate all models on the same held-out test set.
Generate plots and explanations from actual computed outputs.
Evaluation metrics
Accuracy: fraction of predictions that are correct
Precision: fraction of predicted positive cases that are truly positive
Recall: fraction of actual positive cases correctly identified
F1-score: harmonic mean of precision and recall
F2-score: F-score that gives more weight to recall
ROC-AUC: ability to rank positive cases above negative cases
Confusion matrix: counts of correct and incorrect predictions by class
For risk-focused metrics, treat Rejected as the positive class. Report the test-set size and method alongside the results.
Do not promise perfect accuracy. If results are unexpectedly high, discuss possible target separability, dataset limitations, and the need for independent validation.
Dataset predictions and downloads
The prediction-results section should clearly distinguish:
Original dataset target (`loan_status`)
Model-predicted status
Predicted probabilities, when available
Whether the prediction matches the known target
Evaluate model quality using the held-out test set. If generating predictions for every row in the dataset, use a clearly documented method such as out-of-fold predictions where appropriate; do not present training-set predictions as independent test results.
CSV exports should be created from the actual computed results and must not overwrite the original dataset.
Responsible use and limitations
This is an educational demonstration, not a lending decision system.
Historical labels may contain bias.
A model can learn patterns that do not generalize to new applicants or real-world lending data.
Feature importance and SHAP values describe model behavior; they do not establish causation.
Do not display fake metrics, sample applicants, simulated probabilities, or invented explanations as real outputs.
If training or explanation fails, display a clear error or unavailable state instead.
Troubleshooting
Dataset not found
Confirm the file is named `loan_approval_dataset.csv` and is located in `backend/data/`.
Missing Python package
Activate the backend virtual environment and run:
```bash
pip install -r requirements.txt
```
Frontend dependency or script error
From `frontend/`, run:
```bash
npm install
npm run dev
```
Frontend cannot reach the API
Confirm Flask is running, check the API base URL and CORS configuration, and inspect the browser console and backend terminal.
Model or SHAP error
Check package compatibility and the backend error logs. The website should report the error rather than substitute fabricated results.
Project status
Update this section as implementation progresses:
[ ] Dataset validation and summary
[ ] Dataset explorer and CSV download
[ ] Model training and persistence
[ ] Individual applicant predictions
[ ] Test-set model evaluation
[ ] Dataset prediction output and CSV export
[ ] EDA charts
[ ] SHAP explanations
[ ] Frontend and API testing
---
Reminder: Document the results actually produced by your implementation. Do not copy expected scores into the website unless the current run independently calculates those same values.
