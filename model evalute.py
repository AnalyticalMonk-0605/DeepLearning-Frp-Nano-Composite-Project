# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 20:14:09 2026

@author: sanja
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model

# -------- CONFIG --------
DATA_FILE = 'frp_fully_extended.csv'
MODEL_FILE = 'frp_rnn_model_optimized.h5'
SEQUENCE_LENGTH = 5

# -------- LOAD DATA --------
df = pd.read_csv(DATA_FILE)

X_raw = df[['Nano Silica %']].values
y_raw = df[['Tensile Stress (MPa)', 'Flexural Stress (MPa)']].values

# -------- CREATE SEQUENCES --------
X_seq, y_seq = [], []
for i in range(len(X_raw) - SEQUENCE_LENGTH):
    X_seq.append(X_raw[i:i+SEQUENCE_LENGTH])
    y_seq.append(y_raw[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# -------- SCALE DATA (same logic as training) --------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_flat = X_seq.reshape(-1, 1)
X_scaled = scaler_X.fit_transform(X_flat).reshape(X_seq.shape)

y_scaled = scaler_y.fit_transform(y_seq)

# -------- LOAD MODEL --------
model = load_model(
    'frp_rnn_model_optimized.h5',
    compile=False
)

# -------- PREDICT --------
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# -------- METRICS --------
r2_tensile = r2_score(y_seq[:, 0], y_pred[:, 0])
r2_flexural = r2_score(y_seq[:, 1], y_pred[:, 1])

mape_tensile = mean_absolute_percentage_error(y_seq[:, 0], y_pred[:, 0]) * 100
mape_flexural = mean_absolute_percentage_error(y_seq[:, 1], y_pred[:, 1]) * 100

# -------- PRINT RESULTS --------
print("\nMODEL PERFORMANCE (PERCENTAGE VIEW)")
print("=" * 45)

print(f"Tensile Stress Explained Variance (R²): {r2_tensile*100:.2f}%")
print(f"Flexural Stress Explained Variance (R²): {r2_flexural*100:.2f}%")

print("-" * 45)

print(f"Tensile Stress Error (MAPE): {mape_tensile:.2f}%")
print(f"Flexural Stress Error (MAPE): {mape_flexural:.2f}%")

print("=" * 45)
