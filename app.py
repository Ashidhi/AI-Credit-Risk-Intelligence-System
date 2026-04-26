
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
model = joblib.load("models/balanced_random_forest_credit_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="AI Credit Risk Intelligence System", layout="wide")

st.title("AI Credit Risk Intelligence System")
st.write("Loan default risk prediction with cost-sensitive banking intelligence.")

st.sidebar.header("Applicant Details")

duration_months = st.sidebar.slider("Loan Duration (months)", 6, 72, 24)
credit_amount = st.sidebar.number_input("Credit Amount", min_value=250, max_value=20000, value=3000)
age = st.sidebar.slider("Applicant Age", 18, 75, 35)

installment_rate = st.sidebar.slider("Installment Rate", 1, 4, 2)
present_residence_since = st.sidebar.slider("Present Residence Since", 1, 4, 2)
existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 1)
people_liable = st.sidebar.slider("People Liable", 1, 2, 1)

checking_liquidity_score = st.sidebar.slider("Checking Liquidity Score", 0, 3, 1)
savings_strength_score = st.sidebar.slider("Savings Strength Score", 0, 3, 1)
employment_strength_score = st.sidebar.slider("Employment Strength Score", 0, 4, 2)
asset_strength_score = st.sidebar.slider("Asset Strength Score", 0, 3, 1)

threshold = st.sidebar.slider("Risk Threshold", 0.10, 0.90, 0.30, 0.05)

loan_amount_per_month = credit_amount / duration_months

financial_stability_index = (
    employment_strength_score +
    savings_strength_score +
    checking_liquidity_score +
    asset_strength_score
)

repayment_pressure_score = (installment_rate * duration_months) / max(financial_stability_index, 1)

is_high_amount = int(credit_amount > 4000)
is_long_duration = int(duration_months > 24)
is_low_liquidity = int(checking_liquidity_score <= 1)
is_low_savings = int(savings_strength_score == 0)

input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0

manual_values = {
    "duration_months": duration_months,
    "credit_amount": credit_amount,
    "installment_rate": installment_rate,
    "present_residence_since": present_residence_since,
    "age": age,
    "existing_credits": existing_credits,
    "people_liable": people_liable,
    "loan_amount_per_month": loan_amount_per_month,
    "employment_strength_score": employment_strength_score,
    "savings_strength_score": savings_strength_score,
    "checking_liquidity_score": checking_liquidity_score,
    "asset_strength_score": asset_strength_score,
    "financial_stability_index": financial_stability_index,
    "repayment_pressure_score": repayment_pressure_score,
    "is_high_amount": is_high_amount,
    "is_long_duration": is_long_duration,
    "is_low_liquidity": is_low_liquidity,
    "is_low_savings": is_low_savings
}

for col, val in manual_values.items():
    if col in input_data.columns:
        input_data[col] = val

risk_probability = model.predict_proba(input_data)[0][1]
decision = "High Risk / Review Required" if risk_probability >= threshold else "Low Risk / Likely Approve"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Default Risk", f"{risk_probability:.2%}")

with col2:
    st.metric("Decision Threshold", f"{threshold:.2f}")

with col3:
    st.metric("Decision", decision)

profile = pd.DataFrame({
    "Metric": [
        "Loan Amount per Month",
        "Financial Stability Index",
        "Repayment Pressure Score",
        "High Amount Flag",
        "Long Duration Flag",
        "Low Liquidity Flag",
        "Low Savings Flag"
    ],
    "Value": [
        round(loan_amount_per_month,2),
        financial_stability_index,
        round(repayment_pressure_score,2),
        is_high_amount,
        is_long_duration,
        is_low_liquidity,
        is_low_savings
    ]
})

st.dataframe(profile, width='stretch')
