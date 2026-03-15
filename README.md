          Gray-Mobility-AI-ML-Engineer-Intern-Assignment

Smart Ambulance Patient Monitoring System

Developed a smart ambulance AI system that analyzes real-time patient vitals (heart rate, SpO₂, blood pressure) to detect anomalies and assess risk levels. The system includes synthetic time-series data generation, artifact detection, anomaly detection using Isolation Forest, and a FastAPI service for real-time alerts.

1. Introduction

Emergency medical transport environments such as ambulances present unique challenges for patient monitoring systems. Vital signals collected during transport are often affected by noise, motion artifacts, and intermittent sensor failures. Traditional threshold-based alert systems frequently produce false alarms or miss early signs of patient deterioration.

This project develops a machine learning–based monitoring system for a smart ambulance environment. The system processes time-series physiological signals such as heart rate, oxygen saturation (SpO₂), blood pressure, and motion signals. Using artifact detection, feature engineering, and anomaly detection techniques, the system identifies potential early warning signs of patient deterioration and generates risk scores that can assist medical personnel during emergency transport.

The primary objective is to design a robust pipeline capable of handling noisy physiological data and providing meaningful decision support rather than relying solely on simple threshold alerts.

2. Data Generation

Since real ambulance physiological data is difficult to access due to privacy and regulatory constraints, synthetic time-series data was generated to simulate realistic patient monitoring conditions.

The dataset simulates 30 minutes of continuous patient vitals sampled every second. The generated signals include:

  * Heart Rate (HR)
  * Oxygen Saturation (SpO₂)
  * Systolic Blood Pressure
  * Diastolic Blood Pressure
  * Motion/Vibration signals

The dataset contains three types of scenarios:

  1. Normal Transport: Stable vital signs within normal physiological ranges.
  2. Distress Scenarios: Simulated clinical deterioration such as gradual oxygen saturation drops and blood pressure decreases.
  3. Sensor Artifacts: Sudden signal distortions caused by ambulance motion, patient movement, or sensor displacement.

Missing data segments were also introduced to simulate real-world sensor interruptions.

3. Artifact Detection and Signal Cleaning

Physiological sensors in a moving ambulance are highly susceptible to motion artifacts. To prevent false alerts, the system first identifies and handles abnormal signal spikes caused by motion rather than real clinical events.

Artifact detection is performed using rule-based logic that considers:

  * Sudden motion spikes
  * Abrupt SpO₂ drops
  * Sudden heart rate spikes

If a motion spike coincides with an abnormal vital change, the reading is flagged as an artifact. Detected artifact values are then corrected using interpolation and rolling median smoothing techniques. This preprocessing step ensures that the anomaly detection model operates on cleaner and more reliable signals.

4. Feature Engineering

Raw physiological signals are transformed into statistical and temporal features that better represent patient health patterns. A sliding window of 30 seconds is used to compute rolling statistics.

Key engineered features include:

  * Mean heart rate over the window
  * Heart rate variability (standard deviation)
  * Mean SpO₂ level
  * SpO₂ trend indicating oxygen decline
  * Mean systolic blood pressure
  * Blood pressure drop rate
  * Average motion intensity

These features capture both the current state and short-term trends of vital signals, enabling the model to detect subtle physiological deterioration patterns.

5. Anomaly Detection Model

An anomaly detection model was implemented using Isolation Forest. Isolation Forest is well-suited for detecting rare abnormal patterns in high-dimensional datasets.

The model is trained on engineered physiological features to identify unusual patterns in patient vitals. Instead of relying solely on static thresholds, the model detects deviations from normal physiological behavior.

For each observation, the model produces:

  * An anomaly score
  * A binary anomaly flag indicating abnormal conditions

Higher anomaly scores indicate stronger deviations from normal physiological patterns.

6. Risk Scoring Logic

To make anomaly detection outputs clinically meaningful, a rule-based risk scoring system was implemented. The risk score combines multiple vital indicators including:

  * Low oxygen saturation
  * Elevated heart rate
  * Low blood pressure
  * Model-detected anomalies

Each condition contributes to an overall risk score that categorizes patient status into three levels:

    Low Risk – Stable patient condition
    Medium Risk – Potential deterioration requiring monitoring
    High Risk – Critical condition requiring immediate attention

This approach ensures that alerts are interpretable and clinically relevant for medical personnel.

7. API Service for Real-Time Monitoring

To simulate real-time deployment, the anomaly detection model was exposed through an API service using FastAPI. The API accepts incoming patient vitals and returns:

    * Anomaly detection result
    * Risk score
    * Risk level
    * Confidence score

This architecture demonstrates how the system could be integrated into a smart ambulance monitoring platform.

8. Alert Quality Evaluation

The performance of the system can be evaluated using several metrics:

  1. Precision: Percentage of alerts that correspond to true deterioration events.
  2. Recall: Ability of the system to detect actual patient deterioration.
  3. False Alert Rate: Frequency of unnecessary alerts.
  4. Alert Latency: Time difference between physiological deterioration and generated alerts.

In emergency healthcare settings, minimizing missed detections (false negatives) is often more critical than eliminating all false alerts.

9. Failure Analysis

Several potential failure scenarios were analyzed:

  1. Motion artifacts may resemble physiological abnormalities, causing false alerts.
  2. Gradual patient deterioration may not produce strong anomaly signals immediately.
  3. Sensor data loss may reduce model reliability.

Future improvements could include multi-patient training data, more advanced time-series models, and adaptive artifact detection techniques.

10. Safety Considerations

AI systems in medical environments must prioritize safety and transparency. The system is designed as a decision-support tool rather than a fully autonomous diagnostic system. Medical professionals must always remain responsible for final decisions regarding patient treatment.

       This project demonstrates the design of an intelligent patient monitoring system for smart ambulance environments. By combining artifact detection, feature engineering, anomaly detection, and risk scoring, the system provides a more robust alternative to traditional threshold-based alerts.The implementation highlights the importance of handling noisy time-series data and designing interpretable AI systems for safety-critical healthcare applications. Future improvements could include more advanced machine learning models, larger physiological datasets, and integration with real-time ambulance telemetry systems.

