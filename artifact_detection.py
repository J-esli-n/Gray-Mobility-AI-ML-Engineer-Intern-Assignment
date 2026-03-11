import pandas as pd
import numpy as np

df = pd.read_csv("synthetic_ambulance_vitals.csv")

MOTION_THRESHOLD = 0.6
SPO2_DROP_THRESHOLD = 8
HR_SPIKE_THRESHOLD = 25

# compute differences
df["spo2_diff"] = df["spo2"].shift(1) - df["spo2"]
df["hr_diff"] = df["heart_rate"] - df["heart_rate"].shift(1)

# detect conditions
motion_spike = df["motion"] > MOTION_THRESHOLD
spo2_drop = df["spo2_diff"] > SPO2_DROP_THRESHOLD
hr_spike = df["hr_diff"] > HR_SPIKE_THRESHOLD

# artifact detection rule
df["artifact"] = motion_spike & (spo2_drop | hr_spike)

print(df.head())
print("Total artifacts detected:", df["artifact"].sum())
df.to_csv("vitals_with_artifacts_flagged.csv", index=False)