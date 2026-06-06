import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import plotly.express as px
import pickle

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Dynamic pathing based on current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "attention_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
# If you have a tokenizer pickle file, define its path here:
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl") 

# =====================================================
# HEADER
# =====================================================
st.title("🚨 Deep Learning Fraud Detection Dashboard")
st.markdown("Upload transaction data and predict fraud probabilities.")

# =====================================================
# DEBUG INFORMATION (Sidebar)
# =====================================================
with st.sidebar:
    st.subheader("System Information")
    st.write(f"**Root Dir:** `{BASE_DIR}`")
    
    available_files = os.listdir(BASE_DIR)
    st.write("**Available Files in Repository:**")
    st.code(available_files)

# =====================================================
# ARTIFACTS LOADING
# =====================================================
@st.cache_resource
def load_artifacts():
    # 1. Validate and Load Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing. Looked for: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 2. Validate and Load Label Encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file missing. Looked for: {ENCODER_PATH}")
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
        
    # 3. Validate and Load Tokenizer (Optional/Required based on your architecture)
    tokenizer = None
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_artifacts()
    st.sidebar.success("All Artifacts Loaded Successfully!")
except Exception as e:
    st.error(f"Critical Loading Error: {e}")
    st.info("Please verify that 'attention_model.keras' and 'label_encoder.pkl' are uploaded directly into your repository folder.")
    st.stop()

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload CSV File for Analysis", type=["csv"])

# =====================================================
# MAIN APP LOGIC
# =====================================================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Prepare Numeric Data
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.error("Dataset contains no numeric columns for prediction.")
            st.stop()

        # Match Model Expectations
        input_shape = model.input_shape 
        sequence_length = input_shape[1] if input_shape[1] is not None else 1
        expected_features = input_shape[2]

        if numeric_df.shape[1] < expected_features:
            st.error(f"Feature Mismatch: Model needs {expected_features} features, but CSV only has {numeric_df.shape[1]} numeric columns.")
            st.stop()
        
        input_data = numeric_df.iloc[:, :expected_features].values

        if len(input_data) < sequence_length:
            st.error(f"Data too short. Need at least {sequence_length} rows for this model.")
            st.stop()

        X = []
        for i in range(len(input_data) - sequence_length + 1):
            X.append(input_data[i : i + sequence_length])
        
        X = np.array(X)

        # Predictions
        with st.spinner("Analyzing patterns..."):
            predictions = model.predict(X, verbose=0)
            predictions = predictions.flatten()

        # Mapping Results back
        results = df.iloc[sequence_length - 1 :].copy()
        results["Fraud_Probability"] = predictions

        def classify(prob):
            if prob >= 0.8: return "High Risk"
            elif prob >= 0.5: return "Medium Risk"
            return "Low Risk"

        results["Risk_Level"] = results["Fraud_Probability"].apply(classify)

        # UI Visualizations
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyzed", len(results))
        m2.metric("High Risk Found", len(results[results["Risk_Level"] == "High Risk"]))
        m3.metric("Avg Fraud Score", f"{results['Fraud_Probability'].mean():.4f}")

        # Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Trend Analysis")
            fig_trend = px.line(results, y="Fraud_Probability", color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("Risk Distribution")
            fig_pie = px.pie(results, names="Risk_Level", hole=0.4,
                             color="Risk_Level",
                             color_discrete_map={"Low Risk":"green", "Medium Risk":"orange", "High Risk":"red"})
            st.plotly_chart(fig_pie, use_container_width=True)

        # High Risk Table
        st.subheader("🚨 Flagged Transactions")
        st.dataframe(results[results["Risk_Level"] != "Low Risk"].sort_values("Fraud_Probability", ascending=False))

        # Download
        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button("📩 Download Full Report", csv_data, "analysis_results.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.exception(e)

else:
    st.info("👋 Welcome! Please upload a CSV file to begin.")