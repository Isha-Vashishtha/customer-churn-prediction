import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Step 1: Load trained model, scaler, and feature columns ----
model = joblib.load("models/xgb_churn_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

st.title("📊 Telco Customer Churn Prediction App")
st.write("Enter customer details to predict whether the customer will churn.")

# ---- Handle feature names ----
if hasattr(scaler, "feature_names_in_"):
    feature_columns = list(scaler.feature_names_in_)
else:
    feature_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

# Remove unwanted columns if present
drop_cols = ['customerID', 'Churn_Encoded', 'Churn']
feature_columns = [col for col in feature_columns if col not in drop_cols]

# ---- Step 2: Sidebar Inputs ----
st.sidebar.header("🧾 Customer Details")

def get_user_input():
    data = {}
    data['gender'] = 0 if st.sidebar.selectbox('Gender', ['Female', 'Male']) == 'Female' else 1
    data['SeniorCitizen'] = st.sidebar.selectbox('Senior Citizen', [0, 1])
    data['Partner'] = 0 if st.sidebar.selectbox('Partner', ['No', 'Yes']) == 'No' else 1
    data['Dependents'] = 0 if st.sidebar.selectbox('Dependents', ['No', 'Yes']) == 'No' else 1
    data['tenure'] = st.sidebar.number_input('Tenure (months)', 0, 100, 12)
    data['PhoneService'] = 0 if st.sidebar.selectbox('Phone Service', ['No', 'Yes']) == 'No' else 1
    data['MultipleLines'] = st.sidebar.selectbox('Multiple Lines', [0, 1, 2])
    data['InternetService'] = st.sidebar.selectbox('Internet Service', [0, 1, 2])
    data['OnlineSecurity'] = st.sidebar.selectbox('Online Security', [0, 1, 2])
    data['OnlineBackup'] = st.sidebar.selectbox('Online Backup', [0, 1, 2])
    data['DeviceProtection'] = st.sidebar.selectbox('Device Protection', [0, 1, 2])
    data['TechSupport'] = st.sidebar.selectbox('Tech Support', [0, 1, 2])
    data['StreamingTV'] = st.sidebar.selectbox('Streaming TV', [0, 1, 2])
    data['StreamingMovies'] = st.sidebar.selectbox('Streaming Movies', [0, 1, 2])
    data['Contract'] = st.sidebar.selectbox('Contract Type', [0, 1, 2])
    data['PaperlessBilling'] = 0 if st.sidebar.selectbox('Paperless Billing', ['No', 'Yes']) == 'No' else 1
    data['PaymentMethod'] = st.sidebar.selectbox('Payment Method', [0, 1, 2, 3])
    data['MonthlyCharges'] = st.sidebar.number_input('Monthly Charges', 0.0, 1000.0, 70.0)
    data['TotalCharges'] = st.sidebar.number_input('Total Charges', 0.0, 10000.0, 1000.0)

    df = pd.DataFrame(data, index=[0])
    # Add any missing columns (to match training features)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

input_df = get_user_input()

# ---- Step 3: Scale input ----
input_scaled = scaler.transform(input_df)

# ---- Step 4: Make prediction ----
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# ---- Step 5: Display Results ----
st.subheader("📈 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ The customer is **likely to churn.**")
else:
    st.success("✅ The customer is **not likely to churn.**")

st.write(f"**Probability → No Churn:** {prediction_proba[0][0]:.2f} | **Churn:** {prediction_proba[0][1]:.2f}")

# ---- Step 6: SHAP Explainability ----
if st.checkbox("🔍 Show SHAP Explanation"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    # SHAP summary plot (pass fig explicitly)
    st.subheader("Feature Importance (SHAP Summary)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

    # Individual force plot (optional)
    st.subheader("Individual Prediction Explanation")
    shap.initjs()
    st_shap = shap.force_plot(explainer.expected_value, shap_values[0, :], input_df.iloc[0, :], matplotlib=True)
    fig2, ax2 = plt.subplots()
    shap.force_plot(explainer.expected_value, shap_values[0, :], input_df.iloc[0, :], matplotlib=True, show=False)
    st.pyplot(fig2)
