import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load saved objects
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

THRESHOLD = 0.35

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="centered"
)

st.title("🏦 Creditworthiness Prediction App")
st.write("Predict whether a customer is **High Risk** or **Low Risk**")

# -----------------------------
# User Inputs
# -----------------------------
duration = st.number_input("Loan Duration (months)", 1, 72, 24)
credit_amount = st.number_input("Credit Amount", 500, 20000, 5000)
age = st.number_input("Age", 18, 75, 35)
installment_rate = st.slider("Installment Rate (%)", 1, 4, 2)
existing_credits = st.selectbox("Existing Credits", [1, 2, 3, 4])

num_people_liable = st.selectbox("Number of People Liable", [1, 2])
present_residence_since = st.slider(
    "Years at Current Residence", 1, 4, 2
)

housing = st.selectbox("Housing", ["own", "rent", "free"])
job = st.selectbox("Job Type", ["unskilled", "skilled", "management"])
savings = st.selectbox(
    "Savings Account", ["low", "medium", "high", "none"]
)

# -----------------------------
# Create Input Data
# -----------------------------
input_data = {
    'duration_in_month': duration,
    'credit_amount': credit_amount,
    'age': age,
    'installment_rate': installment_rate,
    'existing_credits': existing_credits,
    'num_people_liable': num_people_liable,
    'present_residence_since': present_residence_since,
    'housing_' + housing: 1,
    'job_' + job: 1,
    'savings_account_' + savings: 1
}

input_df = pd.DataFrame([input_data])

# Align columns with training data
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Scale Numerical Features
# -----------------------------
numerical_features = scaler.feature_names_in_

input_df[numerical_features] = scaler.transform(
    input_df[numerical_features]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Credit Risk"):
    risk_prob = model.predict_proba(input_df)[0][1]

    if risk_prob >= THRESHOLD:
        st.error(
            f"❌ **High Credit Risk**\n\n"
            f"Risk Probability: **{risk_prob:.2f}**"
        )
    else:
        st.success(
            f"✅ **Low Credit Risk**\n\n"
            f"Risk Probability: **{risk_prob:.2f}**"
        )
