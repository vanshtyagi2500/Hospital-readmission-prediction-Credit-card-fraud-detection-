# Machine Learning Case Studies

This repository contains two machine-learning case studies.

## 1. Hospital Readmission Prediction
- Logistic Regression
- L2 regularization
- Patient-record features
- ROC-AUC
- Confusion matrix
- Clinical false-negative vs false-positive discussion

## 2. Credit Card Fraud Detection
- XGBoost
- Imbalanced transaction data
- SMOTE
- Decision-threshold tuning
- ROC-AUC and PR-AUC
- Precision, recall and F1
- Feature importance

## Installation

Python 3.10+ is recommended.

Install the dependencies for each case study:

```bash
cd case_study_1_hospital
pip install -r requirements.txt
python hospital_readmission.py
```

and:

```bash
cd ../case_study_2_fraud
pip install -r requirements.txt
python fraud_detection.py
```

Both scripts include a synthetic demonstration dataset so they can be run immediately. For a real project, provide the dataset using `--data`.

## Dataset Safety
Do not upload private patient records, personally identifiable information, payment information, or other sensitive data to a public GitHub repository.

## Disclaimer
These examples are educational machine-learning projects. The hospital model is not a clinical decision-making system, and the fraud model is not a production financial-security system.
