import numpy as np
import pandas as pd

# load cleaned vitals
df_clean = pd.read_csv("synthetic_ambulance_vitals.csv")

# handle missing values
df_clean["spo2"] = df_clean["spo2"].interpolate()

window = 30

features = pd.DataFrame()

features["time_sec"] = df_clean["time_sec"]

features["hr_mean"] = df_clean["heart_rate"].rolling(window).mean()
features["hr_std"] = df_clean["heart_rate"].rolling(window).std()

features["spo2_mean"] = df_clean["spo2"].rolling(window).mean()

features["spo2_trend"] = df_clean["spo2"].rolling(window).apply(
    lambda x: x.iloc[-1] - x.iloc[0]
)

features["bp_sys_mean"] = df_clean["bp_sys"].rolling(window).mean()

features["bp_drop_rate"] = df_clean["bp_sys"].diff()

features["motion_avg"] = df_clean["motion"].rolling(window).mean()

features = features.dropna()

# save features
features.to_csv("features.csv", index=False)

print("Features saved to features.csv")