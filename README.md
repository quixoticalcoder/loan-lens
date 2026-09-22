# loan-lens

An exploratory machine-learning notebook for binary loan-status classification. The workflow examines applicant metadata, handles missing values and outliers, and fits a logistic regression classifier after SMOTE oversampling. It includes a 614-row CSV with 12 input columns and the `Loan_Status` target.

## Workflow

1. Inspect distributions, missing values, and categorical relationships with loan status.
2. Impute categorical fields with their modes and numeric fields with their medians.
3. Retain rows with applicant income below 25,000, coapplicant income below 12,000, and loan amount below 400, in the dataset's original units.
4. Apply `log` to applicant income and `log1p` to coapplicant income; encode categories and remove `Loan_ID`.
5. Balance classes with SMOTE, then make an 80/20 train/test split using `random_state=42`.
6. Fit scikit-learn's `LogisticRegression` and print training accuracy, test accuracy, and a classification report.

The classifier uses 11 features. `Loan_Status` maps `Y` to `1` and `N` to `0`. This is a notebook experiment; no web interface, serving API, or serialized model is provided.

## Data and files

| File | Purpose |
| --- | --- |
| `loan_prediction.ipynb` | Exploration, preprocessing, model fitting, and example prediction |
| `LoanData.csv` | Included tabular dataset with 614 observations |
| `requirements.txt` | Python runtime and notebook dependencies |

Inputs cover gender, marital status, dependents, education, self-employment, applicant and coapplicant income, loan amount, loan term, credit history, and property area. `Loan_ID` identifies records and is excluded before fitting. Dataset provenance, units, and reuse terms are not documented in this repository.

## Run locally

Python 3.11 is a suitable starting environment. Dependencies are unpinned; there is no version lock.

```bash
git clone https://github.com/quixoticalcoder/loan-lens.git
cd loan-lens
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. Open `loan_prediction.ipynb`, choose the environment's Python kernel, and run cells in order. The notebook reads `LoanData.csv` relative to the working directory, so launch Jupyter from the repository root. Retained figures are historical outputs; rerun the notebook to regenerate them.

## Evaluation and implementation limits

- **Data leakage:** imputation and SMOTE run before the train/test split. Test data influence preprocessing and synthetic samples; reported scores do not establish performance on an untouched holdout. Split first, fit preprocessing and resampling only on training data, and evaluate on original held-out observations before comparing models.
- **Reproducibility:** the split is seeded, but SMOTE is not. Results can vary between runs.
- **Encoding:** urban and semiurban locations both map to `1`; all nonzero dependent counts also map to `1`. SMOTE operates on these numeric category encodings and may create fractional categorical values.
- **Example prediction:** the final input array uses raw incomes, although training uses logarithmic incomes. It bypasses the preprocessing steps and should not be treated as a validated inference example.
- **Model scope:** features are not standardized, logistic regression uses default settings, and no tuning, cross-validation, calibration, or subgroup evaluation is implemented. Outlier thresholds change the population being evaluated.

The notebook studies recorded loan-status labels; it does not establish creditworthiness or suitability for automated lending decisions.
