import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

# Load your pre-trained model
try:
    model = load_model('/content/best_model.h5')
    model_loaded = True
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model_loaded = False

# Define the prediction function
def predict(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):
    if not model_loaded:
        return "Model not loaded. Please check the logs."

    columns = [
        'age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome'
    ]
    data = [
        age, job, marital, education, default, balance, housing, loan,
        contact, day, month, duration, campaign, pdays, previous, poutcome
    ]
    df = pd.DataFrame([data], columns=columns)
    df_processed = pd.get_dummies(df)
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_processed:
            df_processed[col] = 0
    df_processed = df_processed[model_columns]
    prediction = model.predict(df_processed)[0]
    return "Yes" if prediction == 1 else "No"

# Streamlit app
st.title("Term Deposit Subscription Prediction")

if model_loaded:
    age = st.number_input("Age")
    job = st.selectbox("Job", ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'student', 'housemaid'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox("Default", ['yes', 'no'])
    balance = st.number_input("Balance")
    housing = st.selectbox("Housing Loan", ['yes', 'no'])
    loan = st.selectbox("Personal Loan", ['yes', 'no'])
    contact = st.selectbox("Contact", ['unknown', 'telephone', 'cellular'])
    day = st.number_input("Day")
    month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    duration = st.number_input("Duration")
    campaign = st.number_input("Campaign")
    pdays = st.number_input("Pdays")
    previous = st.number_input("Previous")
    poutcome = st.selectbox("Poutcome", ['unknown', 'other', 'failure', 'success'])
    
    if st.button("Predict"):
        prediction = predict(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
        st.write(f"Subscription Prediction: {prediction}")
