!pip install yfinance plotly --quiet
# 2️⃣ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ======================================================================
# 3️⃣ Configuration
# ======================================================================
STOCK_TICKER = 'AAPL'   # Change to any symbol like 'GOOG', 'MSFT', 'TSLA'
START_DATE = '2010-01-01'
END_DATE = '2023-01-01'
TIME_STEP = 100         # Lookback period
TRAINING_SPLIT_PERCENT = 0.8

# ======================================================================
# 4️⃣ Data Collection
# ======================================================================
print(f"📥 Fetching historical data for {STOCK_TICKER}...")
data = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE)
print("✅ Data downloaded successfully!")


# ======================================================================
# 5️⃣ Feature Selection & Scaling
# ======================================================================
# Use multiple features for better accuracy
features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Train-test split
training_size = int(len(scaled_features) * TRAINING_SPLIT_PERCENT)
train_data = scaled_features[0:training_size, :]
test_data = scaled_features[training_size:, :]



# ======================================================================
# 6️⃣ Create Dataset Function (Multivariate)
# ======================================================================
def create_multifeature_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step)])
        y.append(dataset[i + time_step, 3])  # Close price index = 3
    return np.array(X), np.array(y)

X_train, y_train = create_multifeature_dataset(train_data, TIME_STEP)
X_test, y_test = create_multifeature_dataset(test_data, TIME_STEP)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")



# ======================================================================
# 7️⃣ Build Bidirectional LSTM Model
# ======================================================================
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(TIME_STEP, X_train.shape[2])),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')



# ======================================================================
# 8️⃣ Callbacks
# ======================================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)
]



# ======================================================================
# 9️⃣ Train Model
# ======================================================================
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ======================================================================
# 🔟 Predictions & Evaluation
# ======================================================================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse scaling (only close price column)
close_index = 3

# Prepare arrays for inverse transformation
train_pred_array = np.zeros((train_pred.shape[0], features.shape[1]))
train_pred_array[:, close_index] = train_pred[:, 0]

y_train_actual_array = np.zeros((y_train.shape[0], features.shape[1]))
y_train_actual_array[:, close_index] = y_train

test_pred_array = np.zeros((test_pred.shape[0], features.shape[1]))
test_pred_array[:, close_index] = test_pred[:, 0]

y_test_actual_array = np.zeros((y_test.shape[0], features.shape[1]))
y_test_actual_array[:, close_index] = y_test

train_pred_transformed = scaler.inverse_transform(train_pred_array)[:, close_index]
y_train_actual_transformed = scaler.inverse_transform(y_train_actual_array)[:, close_index]
test_pred_transformed = scaler.inverse_transform(test_pred_array)[:, close_index]
y_test_actual_transformed = scaler.inverse_transform(y_test_actual_array)[:, close_index]

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual_transformed, test_pred_transformed))
mae = mean_absolute_error(y_test_actual_transformed, test_pred_transformed)
mape = mean_absolute_percentage_error(y_test_actual_transformed, test_pred_transformed)
r2 = r2_score(y_test_actual_transformed, test_pred_transformed)


print("\n📊 Model Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R² Score: {r2:.2f}")


# # ======================================================================
# # 1️⃣1️⃣ Visualization (Interactive Plotly Chart)
# # ======================================================================

plt.figure(figsize=(14,5))
plt.plot(y_test_actual_transformed, color='blue', label='Actual Price')
plt.plot(test_pred_transformed, color='red', label='Predicted Price')
plt.title(f'{STOCK_TICKER} Stock Price Prediction (Test Set)')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# ======================================================================
# 1️⃣2️⃣ Predict Next Day's Price
# ======================================================================
last_time_step_data = scaled_features[-TIME_STEP:].reshape(1, TIME_STEP, features.shape[1])
next_day_scaled = model.predict(last_time_step_data)

# Prepare array for inverse transformation with the correct shape
next_day_scaled_array = np.zeros((1, features.shape[1]))
next_day_scaled_array[:, close_index] = next_day_scaled[0, 0]

next_day_price = scaler.inverse_transform(next_day_scaled_array)[:, close_index]

print(f"\n💰 Predicted closing price for the next trading day: ${next_day_price[0]:.2f}")
