# Study Hours to Score Predictor

An interactive **Streamlit** demonstration of polynomial regression for estimating student test scores from study hours.

## Overview

This project reads a CSV dataset of study hours and test scores, trains a degree-two polynomial regression model, and presents an interface for estimating scores from user-entered study hours. It also displays model-fit metrics.

**Author:** Dimakunne Chukwuebuka Francis  
**Repository:** https://github.com/ebukaxyrus/study-score-predictor  
**Portfolio:** https://xyrusportfolio.netlify.app/

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- scikit-learn (`PolynomialFeatures`, `LinearRegression`, `r2_score`, `mean_absolute_error`)
- Matplotlib

## Implemented features

- Load sample data from `student_test_scores.csv`.
- Build degree-two polynomial features from `study_hours`.
- Train a regression model to estimate `test_score`.
- Display R² and mean absolute error (MAE) for model fit.
- Accept study-hours input through a Streamlit form.

## Project structure

| File | Description |
| --- | --- |
| `Study_app.py` | Streamlit interface, model training, predictions and metrics |
| `student_test_scores.csv` | Input dataset |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
git clone https://github.com/ebukaxyrus/study-score-predictor.git
cd study-score-predictor
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run Study_app.py
```

Run the app from the repository root so the CSV can be found.

## Modelling approach

The code transforms the `study_hours` variable using `PolynomialFeatures(degree=2)` and fits a `LinearRegression` estimator. The UI reports R² and MAE computed from the model's predictions on the training dataset.

**Interpretation:** These metrics measure in-sample fit; they are **not independent test-set accuracy**. Before making generalization claims, add a train/test split or cross-validation and report held-out metrics.

## Limitations and next steps

- Predictions are illustrative and do not independently establish causal effects of study time on grades.
- Test performance on unseen data and check extrapolation for unrealistic study-hour inputs.
- Add data-validation tests, reproducible evaluation, version-pinned dependencies, and screenshots.
- Avoid displaying or committing personal student data without authorization.

## Project status

Educational machine-learning demonstration. Source code was reviewed for documentation, but deployment and end-to-end operation have not been independently verified.
