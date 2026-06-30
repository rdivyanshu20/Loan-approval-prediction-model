import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('loan_approval_model.pkl')
    except FileNotFoundError:
        st.error("Model not found! Please run train_model.py first to generate loan_approval_model.pkl.")
        return None

model = load_model()

# Set up the web page
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant's details below to predict whether their loan will be approved.")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=50000, step=1000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150000, step=1000)

with col2:
    credit_history = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)")
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    property_area = st.selectbox("Property Area", options=["Rural", "Semiurban", "Urban"])

# Process the inputs when the user clicks the button
if st.button("Predict Loan Status"):
    if model is not None:
        # Format the property area to match the get_dummies logic from training
        prop_semiurban = 1 if property_area == "Semiurban" else 0
        prop_urban = 1 if property_area == "Urban" else 0
        
        # Create a dataframe for the model
        input_data = pd.DataFrame({
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Credit_History': [credit_history],
            'Dependents': [dependents],
            'Property_Area_Semiurban': [prop_semiurban],
            'Property_Area_Urban': [prop_urban]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.divider()
        if prediction == 1:
            st.success("### 🎉 Prediction: Approved")
            st.write("This applicant meets the criteria for a loan.")
        else:
            st.error("### ❌ Prediction: Denied")
            st.write("This applicant does not meet the criteria for a loan based on the current model.")
