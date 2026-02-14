import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Page Config
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Classification – Assignment 2")

# Load Metrics
metrics_df = pd.read_csv("model/metrics.csv")

# Sidebar
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload TEST dataset (CSV only)",
    type=["csv"]
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    metrics_df["Model"].tolist()
)

# Display Metrics
st.subheader("Evaluation Metrics")
st.dataframe(metrics_df[metrics_df["Model"] == selected_model])

# Load Model & Scaler
model = joblib.load(f"model/{selected_model.replace(' ', '_')}.pkl")
scaler = joblib.load("model/scaler.pkl")

# Dataset Upload Handling
if uploaded_file:
    df_test = pd.read_csv(uploaded_file, sep=None, engine="python")

    if "y" not in df_test.columns:
        st.error("❌ Uploaded dataset must contain 'y' column")
    else:
        df_test["y"] = df_test["y"].map({"yes": 1, "no": 0})
        df_test = pd.get_dummies(df_test, drop_first=True)

        X_test = df_test.drop("y", axis=1)
        y_test = df_test["y"]

        feature_columns = joblib.load("model/feature_columns.pkl")

        X_test = X_test.reindex(columns=feature_columns, fill_value=0)

        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)