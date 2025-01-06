import streamlit as st
import pickle
import json
import pandas as pd
from groq import Groq
import os 

client = Groq(api_key="Your api key")
MODEL = 'llama3-groq-70b-8192-tool-use-preview'

# Define valid options for categorical fields
VALID_HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VALID_LOAN_INTENT = ["MEDICAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "VENTURE", "PERSONAL", "EDUCATION"]
VALID_LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
VALID_DEFAULT = ["Y", "N"]

def load_model():
    try:
        with open('loan_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_loan_eligibility(person_age, person_income, person_home_ownership, person_emp_length,
                        loan_intent, loan_grade, loan_amnt, loan_int_rate,
                        cb_person_default_on_file, cb_person_cred_hist_length, loan_percent_income):
    try:
        model = load_model()
        if model is None:
            return {"error": "Model loading failed"}
        
        user_input = pd.DataFrame([{
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'loan_percent_income': loan_percent_income,
        }])
        
        prediction = model.predict(user_input)[0]
        
        return {
            "eligibility": bool(prediction),
            "input_data": user_input.to_dict(orient='records')[0],
            "message": "Eligible for loan" if prediction else "Not eligible for loan"
        }
    except Exception as e:
        return {"error": f"Failed to process loan eligibility: {str(e)}"}

def show():
    st.title("🤖 Loan Prediction (GenAI Chatbot)")
    st.markdown("""
    Chat with our AI-powered chatbot to get personalized loan predictions. 
    Answer all required questions to receive your loan eligibility assessment.
    """)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": """You are a loan eligibility assistant. Collect all required information one question at a time. 
                Do not make assumptions or fill in values. Ask for clarification if user input is unclear or invalid."""
            },
            {
                "role": "assistant",
                "content": "Hello! I'm your loan eligibility assistant. I'll help you check your loan eligibility by asking some questions. First, what is your age?"
            }
        ]
        st.session_state.collected_data = {}
        st.session_state.current_field = 'person_age'
        st.session_state.prediction_made = False

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Handle user input
    prompt = st.chat_input("Type your message here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    # Handle post-prediction conversation
                    if st.session_state.prediction_made:
                        if prompt.lower() in ['yes', 'y', 'sure', 'start over', 'new']:
                            st.session_state.collected_data = {}
                            st.session_state.current_field = 'person_age'
                            st.session_state.prediction_made = False
                            response = "Great! Let's start a new loan eligibility check. What is your age?"
                        else:
                            messages = [
                                {"role": "system", "content": """You are a loan expert assistant. Answer questions about loans, 
                                lending, credit, and financial matters. Keep responses focused on loan-related topics."""},
                                {"role": "user", "content": prompt}
                            ]
                            llm_response = client.chat.completions.create(
                                model=MODEL,
                                messages=messages,
                                max_tokens=500
                            )
                            response = llm_response.choices[0].message.content
                    else:
                        current_field = st.session_state.current_field
                        response = None

                        # Process user input based on current field
                        if current_field == 'person_age':
                            try:
                                age = int(prompt)
                                if 18 <= age <= 100:
                                    st.session_state.collected_data['person_age'] = age
                                    st.session_state.current_field = 'person_income'
                                    response = "Great! Now, what is your annual income in dollars?"
                                else:
                                    response = "Please enter a valid age between 18 and 100."
                            except ValueError:
                                response = "Please enter a valid numeric age."

                        elif current_field == 'person_income':
                            try:
                                income = int(prompt)
                                if income > 0:
                                    st.session_state.collected_data['person_income'] = income
                                    st.session_state.current_field = 'person_home_ownership'
                                    response = f"What is your home ownership status? Please choose from: {', '.join(VALID_HOME_OWNERSHIP)}"
                                else:
                                    response = "Please enter a valid positive income amount."
                            except ValueError:
                                response = "Please enter a valid numeric income amount."

                        elif current_field == 'person_home_ownership':
                            if prompt.upper() in VALID_HOME_OWNERSHIP:
                                st.session_state.collected_data['person_home_ownership'] = prompt.upper()
                                st.session_state.current_field = 'person_emp_length'
                                response = "How many years have you been employed? (Enter a number)"
                            else:
                                response = f"Please enter a valid home ownership status: {', '.join(VALID_HOME_OWNERSHIP)}"

                        elif current_field == 'person_emp_length':
                            try:
                                emp_length = int(prompt)
                                if 0 <= emp_length <= 50:
                                    st.session_state.collected_data['person_emp_length'] = emp_length
                                    st.session_state.current_field = 'loan_intent'
                                    response = f"What is the purpose of the loan? Please choose from: {', '.join(VALID_LOAN_INTENT)}"
                                else:
                                    response = "Please enter a valid employment length between 0 and 50 years."
                            except ValueError:
                                response = "Please enter a valid number for employment length."

                        elif current_field == 'loan_intent':
                            if prompt.upper() in VALID_LOAN_INTENT:
                                st.session_state.collected_data['loan_intent'] = prompt.upper()
                                st.session_state.current_field = 'loan_grade'
                                response = f"What is the loan grade? Please choose from: {', '.join(VALID_LOAN_GRADE)}"
                            else:
                                response = f"Please enter a valid loan purpose: {', '.join(VALID_LOAN_INTENT)}"

                        elif current_field == 'loan_grade':
                            if prompt.upper() in VALID_LOAN_GRADE:
                                st.session_state.collected_data['loan_grade'] = prompt.upper()
                                st.session_state.current_field = 'loan_amnt'
                                response = "What is the requested loan amount in dollars?"
                            else:
                                response = f"Please enter a valid loan grade: {', '.join(VALID_LOAN_GRADE)}"

                        elif current_field == 'loan_amnt':
                            try:
                                loan_amount = int(prompt)
                                if loan_amount > 0:
                                    st.session_state.collected_data['loan_amnt'] = loan_amount
                                    st.session_state.current_field = 'loan_int_rate'
                                    response = "What is the interest rate of the loan (as a percentage)?"
                                else:
                                    response = "Please enter a valid positive loan amount."
                            except ValueError:
                                response = "Please enter a valid numeric loan amount."

                        elif current_field == 'loan_int_rate':
                            try:
                                interest_rate = float(prompt)
                                if 0 <= interest_rate <= 100:
                                    st.session_state.collected_data['loan_int_rate'] = interest_rate
                                    st.session_state.current_field = 'cb_person_default_on_file'
                                    response = "Do you have any defaults on file? (Y/N)"
                                else:
                                    response = "Please enter a valid interest rate between 0 and 100."
                            except ValueError:
                                response = "Please enter a valid numeric interest rate."

                        elif current_field == 'cb_person_default_on_file':
                            if prompt.upper() in VALID_DEFAULT:
                                st.session_state.collected_data['cb_person_default_on_file'] = prompt.upper()
                                st.session_state.current_field = 'cb_person_cred_hist_length'
                                response = "How many years of credit history do you have?"
                            else:
                                response = "Please enter Y for Yes or N for No."

                        elif current_field == 'cb_person_cred_hist_length':
                            try:
                                credit_history = int(prompt)
                                if 0 <= credit_history <= 60:
                                    st.session_state.collected_data['cb_person_cred_hist_length'] = credit_history
                                    st.session_state.current_field = 'loan_percent_income'
                                    response = "What percentage of your income would this loan represent? (Enter a number between 0 and 100)"
                                else:
                                    response = "Please enter a valid credit history length between 0 and 60 years."
                            except ValueError:
                                response = "Please enter a valid number for credit history length."

                        elif current_field == 'loan_percent_income':
                            try:
                                percent_income = float(prompt)
                                if 0 <= percent_income <= 100:
                                    st.session_state.collected_data['loan_percent_income'] = percent_income
                                    
                                    # Make prediction
                                    result = get_loan_eligibility(**st.session_state.collected_data)
                                    
                                    if "error" not in result:
                                        response = f"""
                                        📊 **Loan Assessment Results**

                                        {'✅ Eligible for Loan' if result['eligibility'] else '❌ Not Eligible for Loan'}

                                        ### **Personal Details:**
                                        - Age: {st.session_state.collected_data['person_age']} years
                                        - Income: ${st.session_state.collected_data['person_income']:,}
                                        - Home Ownership: {st.session_state.collected_data['person_home_ownership']}
                                        - Employment Length: {st.session_state.collected_data['person_emp_length']} years

                                        ### **Loan Details:**
                                        - Loan Amount: ${st.session_state.collected_data['loan_amnt']:,}
                                        - Interest Rate: {st.session_state.collected_data['loan_int_rate']}%
                                        - Loan Purpose: {st.session_state.collected_data['loan_intent']}
                                        - Loan Grade: {st.session_state.collected_data['loan_grade']}

                                        ### **Credit Details:**
                                        - Credit Default on File: {'Yes' if st.session_state.collected_data['cb_person_default_on_file'] == 'Y' else 'No'}
                                        - Credit History Length: {st.session_state.collected_data['cb_person_cred_hist_length']} years
                                        - Loan as % of Income: {st.session_state.collected_data['loan_percent_income']}%

                                        Would you like to check another loan scenario? Say 'yes' to start over, or feel free to ask any loan-related questions!
                                        """
                                        st.session_state.prediction_made = True
                                    else:
                                        response = f"⚠️ Error: {result['error']}"
                                else:
                                    response = "Please enter a valid percentage between 0 and 100."
                            except ValueError:
                                response = "Please enter a valid number for loan percentage of income."

                        else:
                            response = "An error occurred. Please start over."

                    if response:
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    show()
