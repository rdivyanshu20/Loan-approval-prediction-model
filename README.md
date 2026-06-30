# Loan Approval Prediction Model

This repository contains a machine learning pipeline that predicts whether a loan application will be approved based on applicant financial data. 

## Overview
The model uses a `RandomForestClassifier` to evaluate features like income, credit history, and property area to predict loan eligibility. 

## Files
- `train_model.py`: Generates synthetic data, trains the model, evaluates performance, and saves the final artifact.
- `requirements.txt`: Lists the required Python packages.
- `loan_approval_model.pkl`: The serialized, trained machine learning model (generated dynamically after running the script).

## How to Run
1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
