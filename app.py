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

st.title("Bank Marketing Classification App")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("bank-full.csv", sep=";")

df["y"] = df["y"].map({"yes": 1, "no": 0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Model Selection
# ---------------------------
model_name = st.selectbox(
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

# ---------------------------
# Initialize Model
# ---------------------------
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

# ---------------------------
# Train Model
# ---------------------------
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------
# Evaluation Metrics
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Evaluation Metrics")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    "Score": [accuracy, auc, precision, recall, f1, mcc]
})

st.dataframe(metrics_df)

# ---------------------------
# Confusion Matrix
# ---------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)
