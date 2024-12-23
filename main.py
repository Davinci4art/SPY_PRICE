# Required Libraries
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load Data from Yahoo Finance
def load_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data = data.dropna()
    return data

# Preprocessing
def preprocess_data(data):
    features = ['Open', 'High', 'Low', 'Volume', 'MA50', 'MA200', 'Daily_Return']
    target = 'Close'
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest Model
def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    return predictions

# Train XGBoost Model
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb.fit(X_train, y_train)
    predictions = xgb.predict(X_test)
    return predictions

# Train LSTM Model
def train_lstm(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for LSTM
    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
    predictions = model.predict(X_test_scaled).flatten()
    return predictions

# Evaluation
def evaluate_model(y_test, predictions, model_name):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}")
    return mse, mae

# Plot Results
def plot_results(y_test, predictions, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(predictions, label=f'{model_name} Predictions', color='red')
    plt.title(f'{model_name} Predictions vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Step 1: Load Data
    ticker = "SPY"  # SPY ETF
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    data = load_stock_data(ticker, start_date, end_date)
    print(data.head())

    # Step 2: Preprocess Data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 3: Train and Evaluate Models
    # Random Forest
    rf_predictions = train_random_forest(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, rf_predictions, "Random Forest")
    plot_results(y_test, rf_predictions, "Random Forest")

    # XGBoost
    xgb_predictions = train_xgboost(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, xgb_predictions, "XGBoost")
    plot_results(y_test, xgb_predictions, "XGBoost")

    # LSTM
    lstm_predictions = train_lstm(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, lstm_predictions, "LSTM")
    plot_results(y_test, lstm_predictions, "LSTM")