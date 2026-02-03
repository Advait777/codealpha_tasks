import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("heart_disease_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("🫀 Heart Disease Prediction System")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", 1, 120, 55)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250, 130)
chol = st.number_input("Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.5)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])

# Create input DataFrame
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=feature_names)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Heart Disease Detected (Probability: {probability*100:.2f}%)")
