import os
import schedule
import time
import pandas as pd
import json
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
import threading

working_directory = os.path.dirname(os.path.abspath(__file__))
config_file_path = f"{working_directory}/config.json"
config_data = json.load(open(config_file_path))

# Loading the API key
AVAK = config_data["AV_API_KEY"]
ts = TimeSeries(key=AVAK, output_format='pandas')


def get_stock_data(symbol):
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data = data.sort_index()
    return data.tail(252)


def preprocess_data(data, n_steps):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Load stock data and preprocess it
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NFLX']
stock_data = {symbol: get_stock_data(symbol) for symbol in symbols}


def automated_task():
    for symbol in symbols:
        stock_data_selected = stock_data[symbol]['4. close'].values.reshape(-1, 1)
        n_steps = 60
        X, y, scaler = preprocess_data(stock_data_selected, n_steps)

        # Build and train the model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=5)

        # Save the trained model and scaler
        if not os.path.exists('models'):
            os.makedirs('models')
        model.save(f'models/{symbol}_model.h5')
        np.save(f'models/{symbol}_scaler.npy', scaler)

        print(f"Trained model for {symbol} saved.")


# Background task for the scheduler
def run_scheduler():
    schedule.every().day.at("09:00").do(automated_task)
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every 60 seconds


# Start the scheduler in a separate thread
def start_scheduler():
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # This makes sure the thread will exit when the main program exits
    scheduler_thread.start()


# To start the scheduler, call start_scheduler() in the main program or Streamlit app
if __name__ == "__main__":
    start_scheduler()
    print("Scheduler started.")
