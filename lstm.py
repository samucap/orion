import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sqlite3
import boto3
from io import BytesIO
import pickle
from datetime import datetime, timedelta

# MinIO setup (adjust credentials/endpoint)
s3 = boto3.client('s3', endpoint_url='http://minio:9000', aws_access_key_id='your_key', aws_secret_access_key='your_secret')
BUCKET = 'stock-data'
DB_PATH = 'stock_data.db'

def load_data(ticker, days=60):
    obj = s3.get_object(Bucket=BUCKET, Key=f'historical/{ticker}.parquet')
    df = pd.read_parquet(BytesIO(obj['Body'].read()))
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df.tail(days + 1)  # Extra day for target

def load_scaler(ticker):
    obj = s3.get_object(Bucket=BUCKET, Key=f'scalers/{ticker}_scaler.pkl')
    return pickle.load(BytesIO(obj['Body'].read()))

def prepare_data(df, seq_length=30):
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal']
    scaler = load_scaler(df['ticker'].iloc[0])
    data = df[features].values
    data_scaled = scaler.transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append((df['close'].iloc[i+seq_length] - df['close'].iloc[i+seq_length-1]) / df['close'].iloc[i+seq_length-1])
    
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(ticker, days=1000):
    df = load_data(ticker, days=days)
    X, y = prepare_data(df)
    
    # Chronological split: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    model.save(f'{ticker}_lstm_model.h5')
    
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss for {ticker}: {test_loss}')

def predict_and_save(ticker, seq_length=30):
    model = tf.keras.models.load_model(f'{ticker}_lstm_model.h5')
    df = load_data(ticker, days=60)
    X, _ = prepare_data(df.tail(60), seq_length=seq_length)
    recent_seq = X[-1:]  # Last sequence
    pred_return = model.predict(recent_seq)[0][0]
    trend = 1 if pred_return > 0 else -1
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    cursor.execute("""
        INSERT OR REPLACE INTO lstm_predictions (ticker, timestamp, lstm_prediction)
        VALUES (?, ?, ?)
    """, (ticker, tomorrow, pred_return))
    conn.commit()
    conn.close()
    return pred_return, trend

# Example usage
# train_model('AAPL')  # Run offline
# pred, trend = predict_and_save('AAPL')  # Daily inference
# print(f'Predicted return: {pred}, Trend: {trend}')
