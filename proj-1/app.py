import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🚨",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🚨 Deep Learning Fraud Detection Dashboard")
st.markdown("Upload transaction data and predict fraud probabilities.")

# =====================================================
# PATH RESOLUTION & DEBUG INFORMATION
# =====================================================

# Get the absolute path of the directory containing this script (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "attention_model.keras")

st.sidebar.subheader("System Information")
st.sidebar.write("Current Working Directory:")
st.sidebar.code(os.getcwd())

st.sidebar.write("App Script Directory:")
st.sidebar.code(BASE_DIR)

try:
    available_files = os.listdir(BASE_DIR)
    st.sidebar.write("Available Files in App Directory:")
    st.sidebar.write(available_files)
except Exception as e:
    st.sidebar.error(f"Could not list directory: {e}")

# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found. Looked for '{MODEL_PATH}' but it does not exist."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

try:
    model = load_model()
    st.sidebar.success("Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Model Loading Error: {e}")
    st.error(f"CRITICAL ERROR: Failed to load the deep learning model. {e}")
    st.stop()

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        # ==========================================
        # KEEP ONLY NUMERIC COLUMNS
        # ==========================================
        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        if numeric_df.empty:
            st.error("Dataset contains no numeric columns.")
            st.stop()

        st.subheader("Numeric Features Used")
        st.write(list(numeric_df.columns))

        # ==========================================
        # MODEL INPUT SHAPE
        # ==========================================
        input_shape = model.input_shape
        
        st.write("Model Input Shape:", input_shape)

        # Handle variations in model input shape formats (e.g., [None, sequence, features] or list)
        if isinstance(input_shape, list):
            actual_shape = input_shape[0]
        else:
            actual_shape = input_shape

        sequence_length = actual_shape[1]
        feature_count = actual_shape[2]

        st.write(f"Sequence Length = {sequence_length}")
        st.write(f"Expected Features = {feature_count}")

        # ==========================================
        # VALIDATE FEATURE COUNT
        # ==========================================
        if numeric_df.shape[1] != feature_count:
            st.error(
                f"Feature mismatch. Model expects {feature_count} features. "
                f"Uploaded CSV has {numeric_df.shape[1]} numeric features."
            )
            st.stop()

        # ==========================================
        # CREATE SEQUENCES
        # ==========================================
        features = numeric_df.values
        X = []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])

        X = np.array(X)

        if len(X) == 0:
            st.error(f"Not enough rows to create sequences. Your file needs more than {sequence_length} rows.")
            st.stop()

        st.write("Generated Sequences Shape:", X.shape)

        # ==========================================
        # PREDICTIONS
        # ==========================================
        with st.spinner("Running deep learning model predictions..."):
            predictions = model.predict(X, verbose=0)
            predictions = predictions.flatten()

        # ==========================================
        # RESULTS
        # ==========================================
        # Align target index dataframe rows with sequence predictions length
        results = df.iloc[sequence_length:].copy().reset_index(drop=True)
        results["Fraud_Probability"] = predictions

        def classify(prob):
            if prob >= 0.8:
                return "High Risk"
            elif prob >= 0.5:
                return "Medium Risk"
            else:
                return "Low Risk"

        results["Risk_Level"] = results["Fraud_Probability"].apply(classify)

        # ==========================================
        # METRICS
        # ==========================================
        st.subheader("Fraud Summary")
        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions Evaluated", len(results))
        col2.metric("High Risk Identified", len(results[results["Risk_Level"] == "High Risk"]))
        col3.metric("Average Fraud Score", f"{results['Fraud_Probability'].mean():.4f}")

        # ==========================================
        # HIGH RISK
        # ==========================================
        st.subheader("🚨 High Risk Transactions")
        high_risk = results[results["Risk_Level"] == "High Risk"]
        
        if not high_risk.empty:
            st.dataframe(high_risk, use_container_width=True)
        else:
            st.info("No high risk anomalies detected in this batch.")

        # ==========================================
        # PROBABILITY TREND
        # ==========================================
        st.subheader("Fraud Probability Trend")
        fig = px.line(
            results,
            y="Fraud_Probability",
            title="Timeline Fraud Index Score"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # RISK DISTRIBUTION
        # ==========================================
        st.subheader("Risk Distribution")
        pie_data = results["Risk_Level"].value_counts().reset_index()
        pie_data.columns = ["Risk", "Count"]

        pie_fig = px.pie(
            pie_data,
            names="Risk",
            values="Count",
            color="Risk",
            color_discrete_map={"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # ==========================================
        # DOWNLOAD
        # ==========================================
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions Report CSV",
            data=csv,
            file_name="fraud_predictions_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Processing Error: {e}")

else:
    st.info("Upload a CSV file to start fraud analysis.")