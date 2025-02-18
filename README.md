# StockSage: An Intelligent Stock Price Predictor

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
