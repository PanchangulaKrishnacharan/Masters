import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Classification App")

# ============================================================
# SIDEBAR SECTION
# ============================================================

st.sidebar.header("⚙️ Model & Testing Options")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file for testing",
    type=["csv"]
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("bank-full.csv", sep=";")

df = load_data()

df["y"] = df["y"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("y", axis=1)
y = df["y"]

feature_columns = X.columns  # Save for alignment

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# INITIALIZE MODEL
# ============================================================

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=2000)

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

elif model_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)

elif model_name == "Naive Bayes":
    model = GaussianNB()

elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=42)

elif model_name == "XGBoost":
    model = XGBClassifier(eval_metric="logloss", random_state=42)

# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# DEFAULT TEST SPLIT EVALUATION
# ============================================================

st.subheader("📈 Evaluation Metrics (Test Split)")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    "Score": [
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]
})

st.dataframe(metrics_df)

# Confusion Matrix (Test Split)
st.subheader("🔍 Confusion Matrix (Test Split)")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)

# ============================================================
# UPLOADED CSV EVALUATION
# ============================================================

if uploaded_file:

    st.subheader("📂 Evaluation on Uploaded File")

    df_upload = pd.read_csv(uploaded_file, sep=None, engine="python")

    if "y" not in df_upload.columns:
        st.error("❌ Uploaded dataset must contain 'y' column")

    else:
        df_upload["y"] = df_upload["y"].map({"yes": 1, "no": 0})
        df_upload = pd.get_dummies(df_upload, drop_first=True)

        X_upload = df_upload.drop("y", axis=1)
        y_upload = df_upload["y"]

        # Align columns safely
        X_upload = X_upload.reindex(columns=feature_columns, fill_value=0)

        X_upload = scaler.transform(X_upload)

        y_pred_upload = model.predict(X_upload)
        y_prob_upload = model.predict_proba(X_upload)[:, 1]

        metrics_upload = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            "Score": [
                accuracy_score(y_upload, y_pred_upload),
                roc_auc_score(y_upload, y_prob_upload),
                precision_score(y_upload, y_pred_upload),
                recall_score(y_upload, y_pred_upload),
                f1_score(y_upload, y_pred_upload),
                matthews_corrcoef(y_upload, y_pred_upload)
            ]
        })

        st.dataframe(metrics_upload)

        st.subheader("🔍 Confusion Matrix (Uploaded File)")

        cm_upload = confusion_matrix(y_upload, y_pred_upload)
        fig2, ax2 = plt.subplots()
        ConfusionMatrixDisplay(cm_upload).plot(ax=ax2)
        st.pyplot(fig2)
