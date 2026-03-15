import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import sys
import joblib

# Load and prepare data (assuming feature_engineering.py has been run)
try:
    # Try to load features from previous processing
    features = pd.read_csv("features.csv")  # Assuming features were saved
    df_clean = pd.read_csv("synthetic_ambulance_vitals.csv")
except FileNotFoundError:
    print("Error: Please run feature_engineering.py first to create features")
    sys.exit(1)

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(features)

# Get predictions and scores before adding columns
anomaly_score = model.decision_function(features)
anomaly_predictions = model.predict(features)

# Now add new columns
features["anomaly_score"] = anomaly_score
features["anomaly_flag"] = anomaly_predictions

# convert output
features["anomaly_flag"] = features["anomaly_flag"].map({1:0, -1:1})

# Save the trained model for app.py to use
joblib.dump(model, "isolation_model.pkl")
print("Model saved as isolation_model.pkl")

# Reset index to align with df_clean for visualization
features_reset = features.reset_index(drop=True)

plt.figure(figsize=(12,5))

plt.plot(df_clean["time_sec"], df_clean["heart_rate"], label="Heart Rate")

anomaly_points = features_reset[features_reset["anomaly_flag"] == 1].index

plt.scatter(
    df_clean.loc[anomaly_points, "time_sec"],
    df_clean.loc[anomaly_points, "heart_rate"],
    color="red",
    label="Anomaly"
)

plt.legend()
plt.title("Detected Anomalies in Heart Rate")
plt.show()

features.to_csv("anomaly_results.csv", index=False)