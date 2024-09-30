import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from model_create import get_stock_data, stock_data,start_scheduler
import pandas as pd

start_scheduler()

st.title("FAANG Stock Prediction")


# Function to load the saved model and scaler
def load_model_and_scaler(symbol):
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(f'models/{symbol}_model.h5')
    scaler = np.load(f'models/{symbol}_scaler.npy', allow_pickle=True).item()
    return model, scaler


def predict_next_days(model, scaled_data, n_steps, days=14):
    predictions = []
    last_n_days = scaled_data[-n_steps:]  # Last 60 days from the scaled data
    current_input = last_n_days.reshape((1, n_steps, scaled_data.shape[1]))

    for _ in range(days):
        pred = model.predict(current_input)[0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

    return predictions


symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NFLX']
selected_stock = st.selectbox('Select a stock:', symbols)

if st.button('Predict'):
    # Load the trained model and scaler
    model, scaler = load_model_and_scaler(selected_stock)

    # Load stock data (this should be updated daily as well)
    stock_data_selected = stock_data[selected_stock]['4. close'].values.reshape(-1, 1)

    # Scale the stock data
    stock_data_scaled = scaler.transform(stock_data_selected)

    # Predict the next 14 days
    predictions_scaled = predict_next_days(model, stock_data_scaled, n_steps=60)
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    # Display the predictions
    st.write(f"Predicted stock prices for {selected_stock} for the next 14 days:")
    st.line_chart(pd.DataFrame(predictions, columns=['Predicted Prices']))
