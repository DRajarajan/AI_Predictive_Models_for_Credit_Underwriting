import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('loan_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions
def predict_loan_eligibility(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app layout
st.title("Loan Eligibility Prediction")

# Collect user inputs for the prediction
age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Income", min_value=0, value=50000)
home_ownership = st.selectbox("Home Ownership", options=['RENT', 'OWN', 'MORTGAGE'])
emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
loan_intent = st.selectbox("Loan Intent", options=['PERSONAL', 'EDUCATION', 'MEDICAL'])
loan_grade = st.selectbox("Loan Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input("Loan Amount", min_value=1000, value=5000)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=100.0, value=12.0)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=100.0, value=10.0)
default_on_file = st.selectbox("Default on File (Yes/No)", options=['Y', 'N'])
cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=3)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [home_ownership],
    'person_emp_length': [emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [default_on_file],
    'cb_person_cred_hist_length': [cred_hist_length]
})

# Button to trigger prediction
if st.button("Predict Loan Eligibility"):
    prediction = predict_loan_eligibility(input_data)
    
    if prediction[0] == 1:
        st.success("You are eligible for the loan!")
    else:
        st.error("You are not eligible for the loan.")
