import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_and_preprocess():
    df = pd.read_csv("bank-full.csv", sep=";")

    df["y"] = df["y"].map({"yes": 1, "no": 0})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("y", axis=1)
    y = df["y"]

    feature_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_columns


def get_model(model_name):

    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000)

    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)

    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)

    elif model_name == "Naive Bayes":
        return GaussianNB()

    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)

    elif model_name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=42)
