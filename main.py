import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and encoder
@st.cache_resource
def load_resources():
    model = load_model('best_model.h5')
    encoder = joblib.load('encoder.pkl')  # Load the pre-trained encoder
    return model, encoder

model, encoder = load_resources()

# Helper function for encoding and prediction
def preprocess_and_predict(model, encoder, customer_data):
    # Encode the text data
    encoded_data = encoder.transform(customer_data)
    
    # Predict outcomes
    predictions = model.predict(encoded_data)
    binary_predictions = (predictions >= 0.5).astype(int)
    label_predictions = ["yes" if pred == 1 else "no" for pred in binary_predictions]
    return label_predictions

# Streamlit App
st.title("Customer Data Prediction App")
st.write("Upload a CSV file with text data to predict customer outcomes.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read uploaded file
        customer_data = pd.read_csv(uploaded_file)
        
        # Ensure data is formatted as expected
        st.write("Preview of uploaded data:")
        st.dataframe(customer_data.head())
        
        # Preprocess and predict outcomes
        st.write("Running predictions...")
        predictions = preprocess_and_predict(model, encoder, customer_data)
        
        # Display predictions
        prediction_df = customer_data.copy()
        prediction_df["Prediction"] = predictions
        st.write("Prediction Results:")
        st.dataframe(prediction_df)
        
        # Save predictions to CSV
        output_file = "predictions.csv"
        prediction_df.to_csv(output_file, index=False)
        
        # Provide download link
        st.download_button(
            label="Download Predictions",
            data=open(output_file, "rb").read(),
            file_name="predictions.csv",
            mime="text/csv",
        )
        st.success("Predictions completed and ready for download!")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.write("This app predicts if a customer will subscribe to bank term deposit or not.")
