import streamlit as st
import pandas as pd
import joblib
import os
import datetime

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
    thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 220, 150)

with col2:
    hba1c = st.number_input("HbA1c Level", 4.0, 14.0, 6.0)
    chol = st.number_input("Cholesterol Level", 100, 400, 200)
    bp = st.number_input("Blood Pressure (trestbps)", 80, 200, 120)
    ca = st.number_input("Number of Major Vessels Colored (ca)", 0, 4, 0)

# Predict Button
if st.button("Predict Risk"):
    try:
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
                "thalach": thalach,
                "oldpeak": oldpeak,
                "ca": ca
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

        # Run prediction
        proba = model.predict_proba(X)[0, 1]
        risk = "🔴 High Risk" if proba >= 0.5 else "🟢 Low Risk"

        # Display prediction
        st.markdown(f"<h3>Prediction: {risk}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>Probability: {proba:.2%}</h4>", unsafe_allow_html=True)

        # Save the prediction to a CSV
        record = {
            "timestamp": datetime.datetime.now(),
            "risk_type": risk_choice,
            "predicted_risk": "High Risk" if proba >= 0.5 else "Low Risk",
            "probability": proba,
            "age": age,
            "bmi": bmi,
            "blood_glucose_level": glucose,
            "hba1c_level": hba1c,
            "cholesterol": chol,
            "blood_pressure": bp,
            "thalach": thalach if risk_choice == "Heart Disease" else None,
            "oldpeak": oldpeak if risk_choice == "Heart Disease" else None,
            "ca": ca if risk_choice == "Heart Disease" else None
        }

        # Save to predictions.csv (append if exists, create if not)
        csv_path = os.path.join(script_dir, "predictions.csv")
        if os.path.exists(csv_path):
            pd.DataFrame([record]).to_csv(csv_path, mode='a', header=False, index=False)
        else:
            pd.DataFrame([record]).to_csv(csv_path, index=False)

    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.code(str(e))
