# app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------
# Load Model
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# -----------------------
# UI Layout
# -----------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to **churn** based on their details.")

st.divider()

# -----------------------
# Input Form
# -----------------------
st.subheader("🧾 Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
support_calls = st.number_input("Support Calls", min_value=0, max_value=50, value=2)
payment_delay = st.number_input("Payment Delay (count)", min_value=0, max_value=30, value=0)
total_spend = st.number_input("Total Spend", min_value=0.0, value=2500.0)
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=15)
last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=30)

# -----------------------
# Prediction
# -----------------------
if st.button("🔍 Predict Churn"):
    input_data = pd.DataFrame([{
        "CustomerID": 0,   # 👈 dummy value
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Contract Length": contract_length,
        "Subscription Type": subscription_type,
        "Usage Frequency": usage_frequency,
        "Last Interaction": last_interaction
    }])

    prediction = pipeline.predict(input_data)[0]

    probability = None
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probability = pipeline.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is **likely to churn**")
    else:
        st.success("✅ Customer is **not likely to churn**")

    if probability is not None:
        st.write(f"**Churn Probability:** `{probability:.2f}`")

st.divider()
st.caption("Built with ❤️ using Streamlit & Machine Learning")
