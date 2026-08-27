# Customer churn prediction

MSc Data Science Individual Project (7PAM2002), Semester C 2025-26.
University of Hertfordshire.

Comparing Logistic Regression, Random Forest and XGBoost on the IBM Telco
Customer Churn dataset to see (a) which predicts churn best and (b) which
customer features drive churn.

## Data

`WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 customers, 21 columns, from
Kaggle. Originally an IBM Cognos Analytics sample. Fictional / anonymised,
so no UH ethical approval was required.

## Files

- `Churn_Prediction_Project.ipynb` — the whole analysis
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset
- `artifacts/churn_models.joblib` — saved trained models (produced when you
  run the notebook end-to-end; used by the viva demo)
- `requirements.txt`
- `README.md`

## Running

```bash
pip install -r requirements.txt
jupyter notebook Churn_Prediction_Project.ipynb
```

Or open in Colab — the load cell falls back to a file picker if the CSV
isn't on disk.

Then Run All. Takes about 5 minutes end to end — hyperparameter tuning is
by far the slowest step. `RANDOM_STATE = 42` throughout so results are
reproducible.

## What's in the notebook

- Data cleaning (`TotalCharges` blanks, target encoding)
- EDA — churn balance, contract type, tenure, monthly charges, correlations
- Preprocessing — one-hot encoding, stratified split, scaling
- Three baseline models with imbalance handling
- Hyperparameter tuning via `RandomizedSearchCV` + 5-fold stratified CV
- Final evaluation on the held-out test set
- Feature importance validated two ways (impurity and permutation)
- Decision-threshold tuning against an explicit cost model
- Bootstrap 95% CIs and paired significance tests on the AUC differences
- Saved models + a `predict_churn()` function for the viva demo
- Summary and conclusions

## Key results

After tuning: LR 0.841, RF 0.843, XGBoost 0.845 test ROC-AUC. Bootstrap
significance test says the differences aren't statistically meaningful.
Under the illustrative cost model in section 12, XGBoost is the only model
that produces positive expected business value at its tuned threshold, but
Logistic Regression is the strongest deployment candidate on
interpretability grounds since the predictive performance is effectively
tied.

Main churn drivers: tenure, contract type, fibre-optic internet, total
charges, electronic-cheque payment.


