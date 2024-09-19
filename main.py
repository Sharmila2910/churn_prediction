import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os
from typing import Dict, Union
#Using bankchurn csv
model = None
scaler = None
label_encoders = None

# Data Loading & Preprocessing
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Drop the 'rownumber' 
    data = data.drop(columns=['rownumber'])

    X = data.drop(columns=['churn'])
    y = data['churn']

    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X[X.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoders, scaler

# Model Training
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    joblib.dump(model, "model.pkl")
    return model

# Loading model
def load_model():
    global model, scaler, label_encoders
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("label_encoders.pkl"):
        print("Loading model and preprocessors from saved files...")
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
    else:
        print("No saved model found. Training a new model...")
        filepath = "C:/Users/sharm/Downloads/Churn_prediction/Bank_Churn.csv"
        X_train, X_test, y_train, y_test, label_encoders, scaler = load_and_preprocess_data(filepath)
        model = train_model(X_train, y_train, X_test, y_test)
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoders, "label_encoders.pkl")

# FastAPI 
app = FastAPI()

class PredictionRequest(BaseModel):
    features: Dict[str, Union[str, float, int]]

@app.on_event("startup")
def startup_event():
    
    load_model()

@app.post("/predict/")
def predict(request: PredictionRequest):
    global model, scaler, label_encoders
    features = request.features
    df = pd.DataFrame([features])

    # Drop 'rownumber' if present
    if 'rownumber' in df.columns:
        df = df.drop(columns=['rownumber'])

    # Encode categorical
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Scale numerical 
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_features] = scaler.transform(df[numeric_features])

    # Predict
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
