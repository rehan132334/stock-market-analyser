import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from data.data_collector import data_collector

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── 1. Collect data ──────────────────────────────────────────────────────────
df = data_collector("Apple Inc.")


# ── 2. Extract & scale the Close price ──────────────────────────────────────
close_prices = df.values.reshape(-1, 1)   # shape: (N, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)  # shape: (N, 1)

# Save scaler so you can inverse-transform predictions later
joblib.dump(scaler, "scaler.pkl")

# ── 3. Create sliding-window sequences ──────────────────────────────────────
WINDOW_SIZE = 30   # matches input_shape=(30, 1)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])   # 30 past timesteps
        y.append(data[i, 0])                      # next day's price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices, WINDOW_SIZE)
# X shape: (samples, 30)  →  reshape to (samples, 30, 1) for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# ── 4. Train / validation split (no shuffle — time series!) ─────────────────
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Training samples : {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

# ── 5. Build model ───────────────────────────────────────────────────────────
def build_model():
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(units=48, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model

model = build_model()
model.summary()

# ── 6. Callbacks ─────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# ── 7. Train ─────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ── 8. Save model ────────────────────────────────────────────────────────────
model.save("lstm_model.keras")
print("Model saved to lstm_model.keras")