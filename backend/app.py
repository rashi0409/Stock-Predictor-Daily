import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend for Matplotlib

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the pre-trained LSTM model
try:
    model = load_model('keras_model.h5')
except:
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(100, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Dummy training placeholder; actual training data setup is required
    # model.fit(x_train, y_train, epochs=50)
    model.save('keras_model.h5')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        stock_symbol = data.get("stockSymbol", "").strip().upper()

        if not stock_symbol:
            return jsonify({"error": "Please provide a stock symbol."}), 400

        # Fetch stock data
        end_date = pd.Timestamp.today().date()
        start_date = end_date - pd.DateOffset(years=5)
        df = yf.download(stock_symbol, start=start_date, end=end_date)

        if df.empty:
            return jsonify({"error": f"No data found for {stock_symbol}."}), 404

        # Prepare data for the LSTM model
        df["MA100"] = df["Close"].rolling(100).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        # Splitting into train and test
        train_data = pd.DataFrame(df["Close"][:int(0.8 * len(df))])
        test_data = pd.DataFrame(df["Close"][int(0.8 * len(df)):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)

        x_train, y_train = [], []
        for i in range(100, train_scaled.shape[0]):
            x_train.append(train_scaled[i - 100:i])
            y_train.append(train_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Prepare test data for predictions
        past_100_days = train_data.tail(100)
        final_data = past_100_days._append(test_data, ignore_index=True)
        input_data = scaler.transform(final_data)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_pred = model.predict(x_test)

        # Reverse scaling
        scale = scaler.scale_
        scale_factor = 1 / scale[0]
        y_test = y_test * scale_factor
        y_pred = y_pred.flatten() * scale_factor

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Subplot 1: Closing Price + Moving Averages
        axes[0].plot(df["Close"], color="red", label="Closing Price")
        axes[0].plot(df["MA100"], color="green", label="100-day MA")
        axes[0].plot(df["MA200"], color="blue", label="200-day MA")
        axes[0].set_title(f"Closing Price and Moving Averages for {stock_symbol}")
        axes[0].legend()

        # Subplot 2: Actual vs Predicted Prices
        axes[1].plot(y_test, color="green", label="Actual Prices")
        axes[1].plot(y_pred, color="blue", label="Predicted Prices")
        axes[1].set_title(f"Actual vs Predicted Prices for {stock_symbol}")
        axes[1].legend()

        # Save plot as image
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
