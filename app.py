from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# load model
try:
    model = joblib.load("isolation_model.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Model not found. Please run anomaly_detection_model.py first.")

class Vitals(BaseModel):

    heart_rate: float
    spo2: float
    bp_sys: float
    bp_dia: float
    motion: float


@app.get("/")
def home():

    return {"message": "Smart Ambulance ML API running"}


@app.post("/predict")
def predict(vitals: Vitals):

    # create feature vector - must match features used in training
    X = np.array([[
        vitals.heart_rate,
        vitals.spo2,
        vitals.bp_sys,
        vitals.bp_dia,
        vitals.motion
    ]])

    anomaly = model.predict(X)[0]

    if anomaly == -1:
        anomaly_flag = 1
    else:
        anomaly_flag = 0

    # simple risk scoring based on vitals
    risk_score = 0.0
    risk_level = "low"
    
    if vitals.spo2 < 90:
        risk_score += 0.3
        risk_level = "high"
    elif vitals.spo2 < 94:
        risk_score += 0.2
        if risk_level == "low":
            risk_level = "medium"
    
    if vitals.heart_rate > 120 or vitals.heart_rate < 50:
        risk_score += 0.3
        risk_level = "high"
    
    if vitals.bp_sys > 160 or vitals.bp_sys < 90:
        risk_score += 0.2
        if risk_level == "low":
            risk_level = "medium"
    
    if anomaly_flag == 1:
        risk_score += 0.2
        risk_level = "high"
    
    risk_score = min(risk_score, 1.0)

    return {
        "anomaly_flag": anomaly_flag,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": abs(model.decision_function(X))[0]
    }