import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os


@st.cache_resource
def load_resources():
    model = load_model('best_model.h5')
    return model

model = load_resources()

# Helper function for prediction
def predict_customer_data(model, customer_data):
    # Ensure input data is a NumPy array of type float32
    customer_data = customer_data.astype(np.float32)
    predictions = model.predict(customer_data)
    binary_predictions = (predictions >= 0.5).astype(int)
    label_predictions = ["yes" if pred == 1 else "no" for pred in binary_predictions]
    return label_predictions

# Streamlit App
st.title("Customer Data Prediction App")
st.write("Upload a CSV file to predict customer outcomes.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read uploaded file
        customer_data = pd.read_csv(uploaded_file)

        # Convert all columns to strings for uniform processing
        customer_data = customer_data.astype(str)

        # One-hot encode all columns
        encoded_data = pd.get_dummies(customer_data)

        # Ensure the encoded data matches the model's expected input format
        st.write("Preview of uploaded and encoded data:")
        st.dataframe(encoded_data.head())

        # Predict outcomes
        st.write("Running predictions...")
        predictions = predict_customer_data(model, encoded_data)

        # Display predictions
        prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
        output_df = pd.concat([customer_data, prediction_df], axis=1)
        st.write("Prediction Results:")
        st.dataframe(output_df)

        # Save predictions to CSV
        output_file = "predictions.csv"
        output_df.to_csv(output_file, index=False)

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
st.write("This app predicts if a customer will subscribe to a bank term deposit or not.")
