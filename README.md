# StockSage: Real-Time Insights & Price Predictions

<img width="2528" height="1696" alt="Gemini Generated Image" src="https://github.com/user-attachments/assets/7b9fc948-1502-48fb-851d-056406cf9aa7" />

## Overview
**StockSage** is an intelligent stock price predictor designed to provide insightful information on the trends of any given stock ticker. By utilizing a deep learning approach, StockSage predicts the adjusted close price of a stock using the Long Short-Term Memory (LSTM) model from Keras. The application is built using **Streamlit** and is intended to display historical stock data, including the opening, closing, adjusted closing prices, and trading volume. It also offers a graphical representation of stock price trends for better visualization.

This repository contains the complete source code for StockSage, along with instructions on how to run the application and contribute to it.

## Key Features

- **Stock Price Prediction**: Predict the adjusted close price of a stock based on historical data.
- **Stock Data Information**: Displays the opening, closing, adjusted closing prices, and trading volume of a stock.
- **Visualizations**: Provides visualizations of stock price trends over time using graphs and charts.
- **LSTM Model**: Built using a Keras LSTM (Long Short-Term Memory) network, ideal for sequential data and time-series forecasting.

## Reason for Choosing LSTM Architecture

The LSTM network is chosen for StockSage because of its ability to handle **sequential data** and capture **long-term dependencies**, which are crucial when predicting stock prices. Stock prices are sequential by nature—each day's price is influenced by the prices of previous days. Traditional Recurrent Neural Networks (RNNs) have limitations in retaining long-term dependencies due to their vanishing gradient problem. However, LSTMs, with their **memory cells** and **gating mechanisms**, can effectively learn and remember long-term patterns and trends, making them ideal for stock price prediction.

In addition to handling short-term fluctuations, LSTMs can also account for **long-term influences** on stock prices, such as macroeconomic trends, market events, and company performance, which are crucial for accurate predictions.

<img width="1490" alt="Screenshot 2025-02-18 at 6 43 28 PM" src="https://github.com/user-attachments/assets/15177d56-693c-4256-86ef-3d76ff07f6ea" />
## How It Works

1. **Data Acquisition**: Stock data is fetched using **yfinance**, a Python library that provides stock market data from Yahoo Finance. The data consists of historical stock prices and trading volume.
2. **Data Preprocessing**: The data is cleaned, normalized, and prepared for training the LSTM model. The dataset includes the stock's opening, closing, adjusted closing, and trading volume.
3. **Model Training**: The LSTM model is trained using the historical data to predict future stock prices. The model learns the temporal dependencies from the data to make predictions on unseen data.
4. **Prediction**: Once the model is trained, the app allows users to input the name of a company, fetch its stock data, and predict the adjusted closing price for a future date.
5. **Visualization**: The app displays visualizations such as graphs of stock price trends, helping users visualize price fluctuations and trends over time.

## Technologies Used

- **Streamlit**: For building and running the interactive web application.
- **Keras**: For building and training the LSTM model.
- **yfinance**: For fetching stock data from Yahoo Finance.
- **Matplotlib** & **Plotly**: For visualizing stock price trends and predictions.

## Installation

### Prerequisites

Ensure you have Python 3.7+ installed on your machine. You can download it from the [official Python website](https://www.python.org/).

### Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/StockSage.git
cd StockSage
```
Install the dependencies using pip
```bash
pip install -r requirements.txt
```

Once the dependencies are installed, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
```
This will launch the app in your browser at http://localhost:8501. You can input a company name to view the stock price prediction and associated visualizations.

## Points of Improvement
1. **Dynamic Stock Search**:
The app currently uses a predefined list of stock tickers, mapped using a hashmap between the company name and its corresponding stock ticker. A future enhancement would be to allow users to dynamically search for stock tickers by entering the company name. This would make the app more versatile, eliminating the need for a static list.

2. **Feature Expansion**:
The current model only predicts the adjusted close price of a stock. However, there are many other valuable features in the dataset (e.g., high, low, volume), which could be predicted. Extending the model to predict other features could provide more comprehensive insights into the stock's performance. Additionally, exploring other financial indicators or advanced features, such as volatility or moving averages, could add significant value.

3. **Model Enhancement**:
The LSTM model could be enhanced by incorporating additional features such as technical indicators, news sentiment analysis, or macro-economic data to improve the accuracy of predictions. Experimenting with more advanced architectures, such as GRU (Gated Recurrent Unit) or transformers, might also yield better results for stock price prediction tasks.

4. **Performance Optimization**:
Currently, the app may experience latency due to the time it takes to fetch data from Yahoo Finance and run predictions on the LSTM model. Future optimizations could involve caching stock data or model inference optimizations to reduce response times.

## Contribution
We welcome contributions to enhance StockSage! If you would like to contribute to this project, follow these steps:

Fork the repository.
Create a new branch for your feature (git checkout -b feature-name).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-name).
Open a pull request with a detailed description of the feature you have implemented or the issue you have fixed.
Please ensure your code adheres to the existing coding style and includes tests where applicable.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
yfinance: For fetching historical stock data.
Keras: For providing a robust framework for building LSTM models.
Streamlit: For creating a user-friendly web interface to visualize stock data and predictions.
Matplotlib & Plotly: For creating interactive visualizations of stock trends.
