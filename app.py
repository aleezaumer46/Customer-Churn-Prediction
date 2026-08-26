import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information below to predict whether the customer "
    "is likely to churn."
)

st.divider()

# Customer information
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    customer_id = st.text_input("Customer ID", value="CUSTOMER001")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=0.1
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=0.1
    )


# Prediction button
if st.button("🔮 Predict Customer Churn", type="primary"):

    # Create input DataFrame using the same column names
    # used during model training
    input_data = pd.DataFrame({
        "customerID": [customer_id],
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
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

    try:
        # Apply the same preprocessing used during training
        processed_data = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Get probability if supported
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_data)[0]
            probability = max(probabilities)

        # Convert prediction to Yes/No
        if isinstance(prediction, str):
            churn = prediction.strip().lower() == "yes"
        else:
            churn = int(prediction) == 1

        st.divider()
        st.subheader("Prediction Result")

        if churn:
            st.error("⚠️ Customer is likely to churn.")

            if probability is not None:
                st.write(
                    f"Prediction confidence: **{probability * 100:.2f}%**"
                )
        else:
            st.success("✅ Customer is unlikely to churn.")

            if probability is not None:
                st.write(
                    f"Prediction confidence: **{probability * 100:.2f}%**"
                )

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)