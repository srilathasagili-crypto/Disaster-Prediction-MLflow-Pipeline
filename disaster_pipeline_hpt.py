import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
import joblib
import time
import os
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

mlflow.set_experiment("DISASTER_ALL_MODELS")

df = pd.read_csv("synthetic_disaster_events_2025.csv")

for col in ["disaster_type", "location", "aid_provided"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

drop_cols = [c for c in ["event_id", "date"] if c in df.columns]
df = df.drop(columns=drop_cols)

X = df.drop("is_major_disaster", axis=1)
y = df["is_major_disaster"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "GaussianNB": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([
            ("Scaler", StandardScaler()),
            ("Model", model)
        ])

        train_start = time.time()

        pipe.fit(X_train, y_train)
        
        

        # IDs
        model_id = id(model)
        scaler_id = id(pipe.named_steps["Scaler"])

        train_time = time.time() - train_start

        train_acc = pipe.score(X_train, y_train)

        test_start = time.time()

        y_pred = pipe.predict(X_test)

        test_time = time.time() - test_start

        test_acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model", name)

        mlflow.log_param("model_id", model_id)
        mlflow.log_param("scaler_id", scaler_id)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("test_time", test_time)
        

        mlflow.sklearn.log_model(pipe, name + "_model")

        print(f"{name} -> Test Accuracy: {test_acc:.4f}")