import pickle
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import streamlit as st

# Load your pre-trained model
try:
    model = load_model('best_model.h5')
    model_columns = model.feature_names_in_
    print("Model loaded successfully.")
except AttributeError:
    print("Error: 'Sequential' object has no attribute 'feature_names_in_'.")
    print("This usually means your model was saved with an older TensorFlow version.")
    print("Try saving the model with a newer version or use a different method to access the feature names.")
    # Example of accessing feature names when they're not available in the model attribute:
    # If your model was saved with an older version, you might have to manually load the feature names
    # For example, if you stored the feature names in a pickle file during model training
    #import pickle
    #with open('feature_names.pkl', 'rb') as f:
    #  model_columns = pickle.load(f)
    exit()  # Or handle the error appropriately
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()


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

    # Align processed DataFrame with model input (add missing columns if any)
    for col in model_columns:
        if col not in df_processed:
            df_processed[col] = 0

    df_processed = df_processed[model_columns]

    # Predict
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
