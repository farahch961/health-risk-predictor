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
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🩺 Proactive Healthcare Risk Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown("### Select the Health Risk to Predict:")
risk_choice = st.selectbox(
    "Health Risk:",
    ["Diabetes", "Heart Disease", "Cancer Risk"]
)

# Use columns for layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    glucose = st.number_input("Glucose Level", 50, 300, 100)

with col2:
    hba1c = st.number_input("HbA1c Level", 4.0, 14.0, 6.0)
    chol = st.number_input("Cholesterol Level", 100, 400, 200)
    bp = st.number_input("Blood Pressure", 80, 200, 120)

# Predict Button
if st.button("Predict Risk"):
    if risk_choice == "Diabetes":
        X = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "blood_glucose_level": glucose,
            "hba1c_level": hba1c
        }])
        model = diabetes_model

    elif risk_choice == "Heart Disease":
        X = pd.DataFrame([{
            "age": age,
            "trestbps": bp,
            "chol": chol,
            "bmi": bmi
        }])
        model = heart_model

    else:  # Cancer Risk
        X = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "blood_glucose_level": glucose,
            "hba1c_level": hba1c
        }])
        model = cancer_model

    proba = model.predict_proba(X)[0, 1]
    risk = "🔴 High Risk" if proba >= 0.5 else "🟢 Low Risk"

    st.markdown(f"<h3>Prediction: {risk}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4>Probability: {proba:.2%}</h4>", unsafe_allow_html=True)
