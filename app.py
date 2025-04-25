import streamlit as st
import pandas as pd
import joblib
import os

# Get the path to the directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load your models
diabetes_model = joblib.load(os.path.join(script_dir, "diabetes_model.pkl"))
heart_model    = joblib.load(os.path.join(script_dir, "heart_model.pkl"))
cancer_model   = joblib.load(os.path.join(script_dir, "cancer_model.pkl"))

# App Title
st.title(" Proactive Healthcare Risk Predictor")

# Select Health Risk
risk_choice = st.selectbox(
    "Select the Health Risk to Predict:",
    ["Diabetes", "Heart Disease", "Cancer Risk"]
)

# Common Inputs
age = st.slider("Age", 0, 100, 30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose Level", 50, 300, 100)
hba1c = st.number_input("HbA1c Level", 4.0, 14.0, 6.0)

# Additional inputs if needed
chol = st.number_input("Cholesterol Level", 100, 400, 200)
bp = st.number_input("Blood Pressure (approx)", 80, 200, 120)

# Predict Button
if st.button("Predict Risk"):
    if risk_choice == "Diabetes":
        # Only select features needed for diabetes model
        X = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "blood_glucose_level": glucose,
            "hba1c_level": hba1c
        }])
        model = diabetes_model

    elif risk_choice == "Heart Disease":
        # Select features needed for heart disease model
        X = pd.DataFrame([{
            "age": age,
            "trestbps": bp,
            "chol": chol,
            "bmi": bmi  # optional, if model uses BMI
        }])
        model = heart_model

    elif risk_choice == "Cancer Risk":
        # Select features needed for cancer risk model
        X = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "blood_glucose_level": glucose,
            "hba1c_level": hba1c
        }])
        model = cancer_model

    # Make Prediction
    proba = model.predict_proba(X)[0,1]
    risk = "High Risk" if proba >= 0.5 else "Low Risk"

    # Show Results
    st.markdown(f"### Prediction: **{risk}**")
    st.markdown(f"### Probability: **{proba:.2%}**")


