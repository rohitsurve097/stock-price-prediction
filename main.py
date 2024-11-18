import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Helper Functions
def get_stock_data(ticker, start_date="2010-01-01", end_date="2024-12-31"):
    st.info(f"Fetching stock data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[["Adj Close"]]


def preprocess_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back : i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict_future(model, data, scaler, steps=30):
    future_predictions = []
    input_data = data[-1]
    for _ in range(steps):
        input_reshaped = input_data.reshape(1, input_data.shape[0], 1)
        predicted = model.predict(input_reshaped)[0, 0]
        future_predictions.append(predicted)
        input_data = np.append(input_data[1:], predicted)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


# Streamlit App
st.title("Stock Price Prediction Dashboard")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., IDEA.NS):", "IDEA.NS")
look_back = st.sidebar.slider("Look Back Period (Days):", 30, 120, 60)
prediction_days = st.sidebar.slider("Prediction Period (Days):", 1, 365, 30)
train_model = st.sidebar.button("Train Model and Predict")

if train_model:
    try:
        # Fetch Stock Data
        stock_data = get_stock_data(ticker)
        st.write("Stock Data Preview:")
        st.write(stock_data.tail())

        # Prepare Data
        data = stock_data.values
        X, y, scaler = preprocess_data(data, look_back)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train-Test Split
        split = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        # Train LSTM Model
        st.info("Training LSTM Model...")
        lstm_model = build_lstm_model((X_train.shape[1], 1))
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict Future Prices
        st.info("Predicting Future Prices...")
        future_predictions = predict_future(lstm_model, X[-1:], scaler, prediction_days)

        # Display Predictions
        st.success("Prediction Complete!")
        st.write(f"Predicted Prices for Next {prediction_days} Days:")
        st.write(pd.DataFrame(future_predictions, columns=["Predicted Price"]))

        # Plot Predictions
        st.line_chart(future_predictions)
    except Exception as e:
        st.error(f"Error: {e}")
