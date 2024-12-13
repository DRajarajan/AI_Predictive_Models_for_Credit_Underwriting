# Loan Eligibility Prediction
This project aims to predict loan eligibility using machine learning models. The model is trained on historical loan data, and the prediction is exposed through a Flask API and a user-friendly Streamlit app. The system predicts whether an individual is eligible for a loan based on features like income, employment length, credit history, and loan-related attributes.


# Features
Loan Eligibility Prediction: Predicts whether an individual is eligible for a loan based on features such as income, employment length, and credit score.
Flask API: A RESTful API that provides a POST endpoint for loan eligibility prediction.
Streamlit App: A simple web interface that allows users to enter their details and receive loan eligibility results.

# Setup Instructions
1. Clone the repository
To get started, clone the project repository to your local machine:


# Create a virtual environment and install the required dependencies:


python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

3. Train the Model
To train the model, run the following command to process the data and save the trained model (loan_model.pkl):

python train_model.py
This will train a RandomForestClassifier on your dataset and save the model for later use.

4. Running the Flask API
The Flask API exposes an endpoint to make loan eligibility predictions. To run the Flask API, use the following command:


python app.py
The API will be available at http://127.0.0.1:5000.

5. Running the Streamlit App (Optional)
If you want to use the Streamlit web interface for loan eligibility prediction, use this command to run the app:

streamlit run app.py
This will open the Streamlit app in your browser where you can input your details and check loan eligibility.

API Endpoint
Endpoint: /predict (POST request)
You can make a POST request to the Flask API to predict loan eligibility. Here's an example:

# URL: http://127.0.0.1:5000/predict

Method: POST

# Request Body (JSON):

json
{
    "person_age": 22,
    "person_income": 59000,
    "person_home_ownership": "RENT",
    "person_emp_length": 123,
    "loan_intent": "PERSONAL",
    "loan_grade": "D",
    "loan_amnt": 35000,
    "loan_int_rate": 16.02,
    "loan_percent_income": 0.59,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 3
}
# Response (JSON):

json
Copy code
{
    "loan_eligibility": "Eligible"
}
# Files Overview

# train_model.py
This script trains the machine learning model using the dataset. It handles missing values, performs data preprocessing, and trains a RandomForestClassifier. The trained model is saved as loan_model.pkl.

# app.py
The Flask API that serves the loan eligibility model. It exposes a /predict endpoint that accepts POST requests containing user details and returns whether the user is eligible for a loan or not.

# AI_Predictive_Models_for_Credit_Underwriting.py
This additional file contains models and functionalities for predictive credit underwriting. It can be used to enhance the loan eligibility prediction process by incorporating more advanced credit scoring models.

# loan_model.pkl
This file contains the trained RandomForestClassifier model that is used by the API for making predictions.

# requirements.txt
This file lists all the Python libraries required to run the project. It includes libraries like Flask, Scikit-learn, pandas, and Streamlit.

# Dependencies
To install the required dependencies, run the following command:


pip install -r requirements.txt

The required dependencies include:

Flask
Scikit-learn
pandas
numpy
streamlit
joblib (for saving and loading the model)
Other standard libraries for data preprocessing and model training
