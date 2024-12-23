# Required Libraries
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from scipy.signal import find_peaks
import seaborn as sns
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
    # Technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    
    # Detect peaks and troughs
    peaks, _ = find_peaks(data['Close'], distance=5)
    troughs, _ = find_peaks(-data['Close'], distance=5)
    data['Pattern_Label'] = 0
    for i in peaks: data.iloc[i, data.columns.get_loc('Pattern_Label')] = 1
    for i in troughs: data.iloc[i, data.columns.get_loc('Pattern_Label')] = 2
    
    data = data.dropna()
    return data

# Preprocessing
def preprocess_data(data):
    features = ['Open', 'High', 'Low', 'Volume', 'MA50', 'MA200', 'Daily_Return', 'RSI', 'MACD', 'Pattern_Label']
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
    # Create sequences for LSTM
    def create_sequences(X, y, time_steps=10):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled)

    # Build enhanced LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Train with early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Make predictions
    predictions_scaled = model.predict(X_test_seq)
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    
    # Add movement direction prediction
    pred_direction = np.where(np.diff(predictions) > 0, 1, -1)
    actual_direction = np.where(np.diff(y_test[10:].values) > 0, 1, -1)
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
    print(f"\nPrice Movement Direction Accuracy: {direction_accuracy:.2f}%")
    
    return predictions[:-1]  # Adjust length to match test set

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

# Live Price Monitoring
def monitor_live_prices(ticker):
    stock = yf.Ticker(ticker)
    live_data = stock.history(period='1d', interval='1m')
    if not live_data.empty:
        current_price = live_data['Close'].iloc[-1]
        print(f"Current {ticker} price: ${current_price:.2f}")
    return live_data

# Main Function
if __name__ == "__main__":
    # Step 1: Load Historical Data
    ticker = "SPY"  # SPY ETF
    start_date = "2023-01-01"
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    data = load_stock_data(ticker, start_date, end_date)
    print("\nHistorical Data Preview:")
    print(data.head())
    
    # Live Price Monitoring
    print("\nLive Price:")
    live_data = monitor_live_prices(ticker)

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
    
    # Continuous Live Price Monitoring
    print("\nStarting live price monitoring (Press Ctrl+C to stop)...")
    try:
        while True:
            live_data = monitor_live_prices(ticker)
            plt.pause(60)  # Update every minute
    except KeyboardInterrupt:
        print("\nLive monitoring stopped")