import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; 

const App = () => {
    const [stockSymbol, setStockSymbol] = useState('');
    const [graphUrl, setGraphUrl] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setGraphUrl(null);

        try {
            // Send the stock symbol to the Flask backend
            const response = await axios.post('http://127.0.0.1:5000/predict', { stockSymbol }, { responseType: 'blob' });
            
            // Create a URL for the plot image returned by Flask
            const blob = new Blob([response.data], { type: 'image/png' });
            const url = URL.createObjectURL(blob);
            setGraphUrl(url);
        } catch (err) {
            if (err.response && err.response.data.error) {
                setError(err.response.data.error);
            } else {
                setError('Please check the stock symbol as per the Yahoo listing and try again :)');
            }
        }
    };

    return (
      <div className="app-container">
        <div className="form-container">
            {/* Add a stock market icon */}
            <div className="icon-container">
                <i className="fas fa-chart-line"></i>
            </div>
            <div>
              <h1>Stock Price Prediction</h1>
            <form onSubmit={handleSubmit}>
                <label htmlFor="stockSymbol">
                    <i className="fas fa-search-dollar"></i> Enter Stock Symbol:
                </label>
                <input
                    type="text"
                    id="stockSymbol"
                    value={stockSymbol}
                    onChange={(e) => setStockSymbol(e.target.value)}
                    placeholder="e.g., GOOGL, AAPL"
                    required
                />
                <button type="submit">
                    <i className="fas fa-chart-pie"></i> Predict
                </button>
            </form>
            </div>

              {error && <p style={{ color: 'black' }}>{error}</p>}
              {graphUrl && (
                  <div className="graph-container">
                      <h2>Prediction Results</h2>
                      <img src={graphUrl} alt="Stock Price Prediction Graph" />
                      <a href="/" className="back-button">Back</a>
                  </div>
              )}
          </div>
        </div>
    );
};

export default App;
