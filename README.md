# StockSage: An Intelligent Stock Price Predictor
StockSage provides insightful information about the trends of a given stock ticker and predicts the adjusted close price for that stock. This repository contains a Streamlit application that, given a company's name, informs you about the opening, closing, adjusted closing prices, and trading volume of the stock, and provides visualizations for stock trends. It has been built using an LSTM model from Keras.

*Reason for choosing such an architecture:* LSTM networks are particularly well-suited for stock price prediction due to their ability to handle sequential data and capture long-term dependencies. Stock prices are inherently sequential, with each day's price influenced by previous days. LSTM networks can excel in such tasks. Moreover, stock prices can be influenced by events and trends that span long periods. LSTMs, with their memory cells and gating mechanisms, can learn and remember these long-term dependencies better than traditional RNNs.


# Points of Improvement
The app is currently limited to a set number of stock tickers since the options are created using a hashmap with the mapping between the company name and its stock ticker. A feature for dynamic searching must be implemented. Additionally, the app only predicts the adjusted close price of the stock, so there is an opportunity to extend its capabilities to predict the value of any feature that appears in the dataset.
