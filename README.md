# Stock Predictor Daily

A full-stack stock prediction web application that leverages **LSTM models**, **Yahoo Finance API**, and **Flask-React integration** to provide short-term price predictions and historical trend visualizations for publicly traded stocks.

## Features

-  Predicts stock prices using Long Short-Term Memory (LSTM) neural networks
-  Fetches up to 5 years of historical stock data from the Yahoo Finance API
-  Achieves prediction accuracy of **60–65%** using trained LSTM models
-  Calculates and displays **100-day and 200-day moving averages**
-  Renders dynamic, interactive stock graphs with **Matplotlib**
-  Frontend built with **React.js** for responsive and animated user experience
-  Real-time interaction between **React frontend** and **Flask backend**

## Tech Stack

- **Frontend**: React.js, HTML5, CSS3, JavaScript
- **Backend**: Flask, Python, TensorFlow, scikit-learn
- **Data & APIs**: Yahoo Finance API
- **Visualization**: Matplotlib
- **Machine Learning**: LSTM (Long Short-Term Memory) Neural Networks
- **Deployment**: Vercel (or compatible full-stack deployment)

## How It Works

1. **User Input**: Enter the stock ticker symbol in the React-based UI.
2. **Data Retrieval**: Flask backend fetches historical stock data using the Yahoo Finance API.
3. **Preprocessing**: Data is normalized and processed for model input.
4. **Prediction**: A pre-trained LSTM model analyzes the data and predicts future prices.
5. **Visualization**: The frontend displays actual vs. predicted prices along with moving averages.
6. **Interactivity**: Graphs and data update in real-time based on user input.

## Key Highlights

- Built a custom LSTM model architecture for time series forecasting
- Implemented moving average strategies for trend tracking
- Connected Flask APIs to React using asynchronous calls for smooth UX
- Designed a sleek, responsive UI with CSS animations and state handling
- Integrated Matplotlib in Flask to dynamically render charts on demand

