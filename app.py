from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

for artifact_path in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH):
    if not artifact_path.exists():
        st.error(f"Missing required model file: {artifact_path.name}")
        st.stop()

knn_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)

st.title("Heart Disease Prediction")
st.markdown("Provide the following details to check your heart disease risk:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    input_df = pd.DataFrame([raw_input])
    encoded_input = pd.get_dummies(input_df, drop_first=True)
    encoded_input = encoded_input.reindex(columns=columns, fill_value=0)

    numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    encoded_input[numerical_cols] = scaler.transform(encoded_input[numerical_cols])
    prediction = int(knn_model.predict(encoded_input)[0])

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
