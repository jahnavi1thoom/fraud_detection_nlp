
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
# DEBUG INFORMATION
# =====================================================

st.sidebar.subheader("System Information")
st.sidebar.write("Current Directory:")
st.sidebar.write(os.getcwd())

available_files = os.listdir()

st.sidebar.write("Available Files:")
st.sidebar.write(available_files)

# =====================================================
# MODEL LOADING
# =====================================================

MODEL_PATH = "attention_model.keras"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found in project folder"
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
    st.error(f"Model Loading Error: {e}")
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
            st.error(
                "Dataset contains no numeric columns."
            )
            st.stop()

        st.subheader("Numeric Features Used")

        st.write(
            list(numeric_df.columns)
        )

        # ==========================================
        # MODEL INPUT SHAPE
        # ==========================================

        input_shape = model.input_shape

        st.write(
            "Model Input Shape:",
            input_shape
        )

        sequence_length = input_shape[1]
        feature_count = input_shape[2]

        st.write(
            f"Sequence Length = {sequence_length}"
        )

        st.write(
            f"Expected Features = {feature_count}"
        )

        # ==========================================
        # VALIDATE FEATURE COUNT
        # ==========================================

        if numeric_df.shape[1] != feature_count:

            st.error(
                f"""
                Feature mismatch.

                Model expects {feature_count} features.

                Uploaded CSV has {numeric_df.shape[1]} features.
                """
            )

            st.stop()

        # ==========================================
        # CREATE SEQUENCES
        # ==========================================

        features = numeric_df.values

        X = []

        for i in range(
            len(features) - sequence_length
        ):

            X.append(
                features[
                    i:i+sequence_length
                ]
            )

        X = np.array(X)

        st.write(
            "Generated Sequences:",
            X.shape
        )

        if len(X) == 0:

            st.error(
                "Not enough rows to create sequences."
            )

            st.stop()

        # ==========================================
        # PREDICTIONS
        # ==========================================

        predictions = model.predict(
            X,
            verbose=0
        )

        predictions = predictions.flatten()

        # ==========================================
        # RESULTS
        # ==========================================

        results = df.iloc[
            sequence_length:
        ].copy()

        results[
            "Fraud_Probability"
        ] = predictions

        def classify(prob):

            if prob >= 0.8:
                return "High Risk"

            elif prob >= 0.5:
                return "Medium Risk"

            else:
                return "Low Risk"

        results["Risk_Level"] = results[
            "Fraud_Probability"
        ].apply(classify)

        # ==========================================
        # METRICS
        # ==========================================

        st.subheader("Fraud Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transactions",
            len(results)
        )

        col2.metric(
            "High Risk",
            len(
                results[
                    results["Risk_Level"]
                    == "High Risk"
                ]
            )
        )

        col3.metric(
            "Average Fraud Score",
            round(
                results[
                    "Fraud_Probability"
                ].mean(),
                4
            )
        )

        # ==========================================
        # HIGH RISK
        # ==========================================

        st.subheader(
            "🚨 High Risk Transactions"
        )

        high_risk = results[
            results["Risk_Level"]
            == "High Risk"
        ]

        st.dataframe(
            high_risk,
            use_container_width=True
        )

        # ==========================================
        # PROBABILITY TREND
        # ==========================================

        st.subheader(
            "Fraud Probability Trend"
        )

        fig = px.line(
            results,
            y="Fraud_Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # RISK DISTRIBUTION
        # ==========================================

        st.subheader(
            "Risk Distribution"
        )

        pie_data = (
            results["Risk_Level"]
            .value_counts()
            .reset_index()
        )

        pie_data.columns = [
            "Risk",
            "Count"
        ]

        pie_fig = px.pie(
            pie_data,
            names="Risk",
            values="Count"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

        # ==========================================
        # DOWNLOAD
        # ==========================================

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(
            f"Processing Error: {e}"
        )

else:

    st.info(
        "Upload a CSV file to start fraud analysis."
    )

