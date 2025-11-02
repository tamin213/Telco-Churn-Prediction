import streamlit as st
import pandas as pd
import joblib

st.title("Telco Customer Churn Prediction")

st.write("""
This app collects customer details to predict whether a customer is likely to churn or not.
Please enter the customer details below
""")

st.header("Customer Information")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
with col2:
    tenure = st.slider("Tenure (in months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

st.header("Internet and Services")
col3, col4 = st.columns(2)
with col3:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
with col4:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.header("Billing and Payment")
col5, col6 = st.columns(2)
with col5:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
with col6:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

st.markdown("---")

if st.button("Predict Churn"):
    model = joblib.load("xgc.joblib")
    encoders = joblib.load("encoders.joblib")
    columns = joblib.load("columns.joblib")

    # Map SeniorCitizen directly to 0/1 instead of strings
    senior_citizen_value = 1 if senior_citizen == "Yes" else 0

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen_value],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Apply label encoders only on categorical columns that were encoded before
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Ensure column order matches training data
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Ensure all columns are numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"The customer is likely to CHURN (Probability: {probability:.2f})")
    else:
        st.success(f"The customer is likely to STAY (Probability: {probability:.2f})")