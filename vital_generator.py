import numpy as np
import pandas as pd

# simulation settings
duration_minutes = 30
sampling_rate = 1  # seconds
total_points = duration_minutes * 60

np.random.seed(42)

time = np.arange(total_points)

# -----------------------------
# NORMAL BASELINE VITALS
# -----------------------------
heart_rate = np.random.normal(80, 4, total_points)
spo2 = np.random.normal(98, 0.5, total_points)
bp_sys = np.random.normal(120, 5, total_points)
bp_dia = np.random.normal(80, 4, total_points)
motion = np.random.normal(0.05, 0.02, total_points)

# -----------------------------
# DISTRESS SCENARIO 1: HYPOXIA
# gradual SpO2 drop
# -----------------------------
start = 600
end = 750

spo2[start:end] -= np.linspace(0, 10, end-start)
heart_rate[start:end] += np.linspace(0, 15, end-start)

# -----------------------------
# DISTRESS SCENARIO 2: SHOCK
# BP drop + HR increase
# -----------------------------
start = 1100
end = 1250

bp_sys[start:end] -= np.linspace(0, 25, end-start)
bp_dia[start:end] -= np.linspace(0, 15, end-start)
heart_rate[start:end] += np.linspace(0, 20, end-start)

# -----------------------------
# SENSOR ARTIFACTS
# motion spikes causing false readings
# -----------------------------
artifact_points = np.random.choice(total_points, 20)

for i in artifact_points:
    motion[i] = np.random.uniform(0.7, 1.0)
    spo2[i] -= np.random.uniform(10, 20)
    heart_rate[i] += np.random.uniform(20, 40)

# -----------------------------
# MISSING SENSOR DATA
# -----------------------------
missing_points = np.random.choice(total_points, 15)

for i in missing_points:
    spo2[i] = np.nan

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "time_sec": time,
    "heart_rate": heart_rate,
    "spo2": spo2,
    "bp_sys": bp_sys,
    "bp_dia": bp_dia,
    "motion": motion
})

# clip physiological limits
df["heart_rate"] = df["heart_rate"].clip(40, 180)
df["spo2"] = df["spo2"].clip(70, 100)
df["bp_sys"] = df["bp_sys"].clip(70, 200)
df["bp_dia"] = df["bp_dia"].clip(40, 130)

# save dataset
df.to_csv("synthetic_ambulance_vitals.csv", index=False)

print("Synthetic ambulance vitals dataset generated!")
print(df.head())