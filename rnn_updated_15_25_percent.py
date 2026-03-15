import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION AND DATA LOADING ---
DATA_FILE = 'frp_fully_extended.csv' 
SEQUENCE_LENGTH = 5  # Increased sequence length to capture more temporal context
HIDDEN_UNITS = [256,128] 
DROPOUT_RATE = 0.3 
BATCH_SIZE = 32 
EPOCHS = 500 
INITIAL_LEARNING_RATE = 0.0001
EXCEL_OUTPUT_FILE = 'frp_rnn_model_results.xlsx' # New Excel output file name

try:
    # Load EXTENDED dataset (includes 15-25% nano silica data)
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found. Please ensure it is in the working directory.")
    exit()

print(f"Total dataset size: {len(df)} samples")
print(f"Nano Silica % range: {df['Nano Silica %'].min():.2f} - {df['Nano Silica %'].max():.2f}")


# --- 1. DATA PREPARATION (Sequential Data Creation) ---

X_raw = df[['Nano Silica %']].values
y_raw = df[['Tensile Stress (MPa)', 'Flexural Stress (MPa)']].values

X_seq, y_seq = [], []
for i in range(len(X_raw) - SEQUENCE_LENGTH):
    X_seq.append(X_raw[i:i+SEQUENCE_LENGTH])
    y_seq.append(y_raw[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


# --- 2. OPTIMIZED SCALING ---

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Flatten train features for scaling then reshape back
X_train_flat = X_train.reshape(-1, 1)
X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)

X_test_flat = X_test.reshape(-1, 1)
X_test_scaled_flat = scaler_X.transform(X_test_flat)
X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)

# Scale targets
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


# --- 3. OPTIMIZED MODEL ARCHITECTURE (GRU and Advanced Regularization) ---

model = Sequential()
model.add(GRU(
    units=HIDDEN_UNITS[0],
    activation='tanh', 
    return_sequences=True,
    recurrent_dropout=DROPOUT_RATE, 
    input_shape=(SEQUENCE_LENGTH, 1)
))

model.add(GRU(
    units=HIDDEN_UNITS[1],
    activation='tanh',
    recurrent_dropout=DROPOUT_RATE
))

model.add(Dropout(DROPOUT_RATE))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2)) 

optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("\nOptimized Model Summary (GRU-based):")
model.summary()


# --- 4. ADVANCED CALLBACKS (Learning Rate Scheduling) ---

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5, 
    patience=20, 
    min_lr=1e-6, 
    verbose=1
)

# Train model
print(f"\nStarting training with SEQUENCE_LENGTH={SEQUENCE_LENGTH}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# --- 5. EVALUATION AND RESULTS GENERATION ---

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate
r2_tensile = r2_score(y_test[:, 0], y_pred[:, 0])
r2_flexural = r2_score(y_test[:, 1], y_pred[:, 1])
rmse_tensile = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
rmse_flexural = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
mae_tensile = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_flexural = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

evaluation_results = {
    'Metric': ['R2_Tensile', 'R2_Flexural', 'RMSE_Tensile', 'RMSE_Flexural', 'MAE_Tensile', 'MAE_Flexural'],
    'Value': [r2_tensile, r2_flexural, rmse_tensile, rmse_flexural, mae_tensile, mae_flexural]
}
df_metrics = pd.DataFrame(evaluation_results) # Convert dict to DataFrame

# Prepare prediction results dataframe
test_results = pd.DataFrame({
    'Nano_Silica_%': X_test[:, -1, 0], 
    'Actual_Tensile_Stress': y_test[:, 0],
    'Predicted_Tensile_Stress': y_pred[:, 0],
    'Actual_Flexural_Stress': y_test[:, 1],
    'Predicted_Flexural_Stress': y_pred[:, 1]
})

# FILTER FOR OPTIMAL RESULTS (Now using 10% < Nano Silica < 20% as per your last filter)
optimal_results = test_results[
    (test_results['Nano_Silica_%'] > 10) & 
    (test_results['Nano_Silica_%'] < 20) &
    (test_results['Predicted_Tensile_Stress'] > 15)
]



# --- 6. EXCEL EXPORT (Consolidating all results) ---

with pd.ExcelWriter(EXCEL_OUTPUT_FILE) as writer:
    test_results.to_excel(writer, sheet_name='Full_Test_Predictions', index=False)
    optimal_results.to_excel(writer, sheet_name='Optimal_Results', index=False)
    df_metrics.to_excel(writer, sheet_name='Evaluation_Metrics', index=False)

print("\n" + "="*60)
print(f"✓ All model results saved to: {EXCEL_OUTPUT_FILE}")
print("="*60)
for key, value in zip(df_metrics['Metric'], df_metrics['Value']):
    print(f"{key}: {value:.4f}")

# Save the trained model for future use
model.save('frp_rnn_model_optimized.h5')
print("\n✓ Trained model saved to: frp_rnn_model_optimized.h5")

