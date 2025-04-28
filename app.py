import streamlit as st
import pandas as pd
import joblib
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import json

# Load your models
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")
cancer_model = joblib.load("cancer_model.pkl")

# App Title
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🩺 Proactive Healthcare Risk Predictor</h1>
""", unsafe_allow_html=True)

st.markdown("### Select the Health Risk to Predict:")
risk_choice = st.selectbox("Health Risk:", ["Diabetes", "Heart Disease", "Cancer Risk"])

# Common Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 0, 100, 30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
with col2:
    hba1c = st.number_input("HbA1c Level", 4.0, 14.0, 6.0)
    chol = st.number_input("Cholesterol Level", 100, 400, 200)
    bp = st.number_input("Blood Pressure", 80, 200, 120)

# Heart-specific inputs
if risk_choice == "Heart Disease":
    st.markdown("### Additional Heart Health Info:")
    col3, col4, col5 = st.columns(3)
    with col3:
        thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 220, 150)
    with col4:
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    with col5:
        ca = st.number_input("Number of Major Vessels Colored (ca)", 0, 4, 0)

# Predict Button
if st.button("Predict Risk"):
    if risk_choice == "Diabetes":
        X = pd.DataFrame([{ "age": age, "bmi": bmi, "blood_glucose_level": glucose, "hba1c_level": hba1c }])
        model = diabetes_model
    elif risk_choice == "Heart Disease":
        X = pd.DataFrame([{ "age": age, "trestbps": bp, "chol": chol, "thalach": thalach, "oldpeak": oldpeak, "ca": ca }])
        model = heart_model
    else:
        X = pd.DataFrame([{ "age": age, "bmi": bmi, "blood_glucose_level": glucose, "hba1c_level": hba1c }])
        model = cancer_model

    # Predict and display result
    proba = model.predict_proba(X)[0, 1]
    risk = "🔴 High Risk" if proba >= 0.5 else "🟢 Low Risk"
    st.markdown(f"<h3>Prediction: {risk}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4>Probability: {proba:.2%}</h4>", unsafe_allow_html=True)

try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    record = {
        "timestamp": pd.Timestamp.now(),
        "risk_type": risk_choice,
        "predicted_risk": "High Risk" if proba >= 0.5 else "Low Risk",
        "probability": float(proba),
        "age": int(age),
        "bmi": float(bmi),
        "blood_glucose_level": int(glucose),
        "hba1c_level": float(hba1c),
        "cholesterol": int(chol),
        "blood_pressure": int(bp),
        "thalach": int(thalach) if risk_choice == "Heart Disease" else None,
        "oldpeak": float(oldpeak) if risk_choice == "Heart Disease" else None,
        "ca": int(ca) if risk_choice == "Heart Disease" else None
    }

    df_record = pd.DataFrame([record])
    table_id = "capstone-project-457819.health_analytics.prediction_logs"
    job = client.load_table_from_dataframe(df_record, table_id)
    job.result()
    st.success("Prediction logged to BigQuery.")
except Exception as e:
    st.warning(f"Failed to log to BigQuery: {e}")
