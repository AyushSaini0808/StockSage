# Importing necessary libraries
import streamlit as st
import time
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import math
# Creating a hashmap to map company name with corresponding stock ticker
company_map={"Google":"GOOG","NVIDIA":"NVDA","Netflix":"NFLX","Amazon":"AMZN","Apple":"AAPL","Tesla":"TSLA","Mircosoft":"MSFT","Meta":"META"}
myKeys = list(company_map.keys())
myKeys.sort()
company_map = {i: company_map[i] for i in myKeys}
# Applying some CSS formatting to enhance the web app
page_bg_img="""
<style>
[data-testid="stAppViewContainer"]{
background: rgb(0,0,0);
background: linear-gradient(90deg, rgba(0,0,0,1) 0%, rgba(71,71,71,1) 35%, rgba(0,0,0,1) 100%);
}
[data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"]{
background: rgb(115,111,111);
color:rgb(0,0,0);
background: linear-gradient(90deg, rgba(115,111,111,1) 0%, rgba(133,133,133,1) 40%, rgba(173,172,172,1) 96%);
</style>
""" 
#Setting initial configurations of the app
st.set_page_config(page_title="StockSage",layout="wide",page_icon="📈",initial_sidebar_state="expanded")
st.markdown(page_bg_img,unsafe_allow_html=True)
# Building the app
title=st.title('StockSage : Real-Time Insights & Price Predictions')
st.divider()
# Creating the sidebar content
st.sidebar.markdown(f"## About the Project:")
st.sidebar.markdown("""### StockSage is designed to provide insightful information regarding the stock trends of some of the major tech. companies. Utilizing LSTM(Long-Short Term Memory),a recurrent neural network, StockSage analyzes historical stock data to forecast future price movements with high accuracy.""")
st.sidebar.divider()
st.sidebar.markdown("Made by **Ayush Saini**")

# The main page
st.sidebar.markdown("## Select the company ticker:")
selected_ticker = st.sidebar.selectbox("", list(company_map.keys()), placeholder="Select a ticker")
ticker_symbol = company_map[selected_ticker]
end = datetime.now()
start = datetime(end.year - 6, end.month, end.day)
data = yf.download(ticker_symbol, start, end)

# Get company info and financials
company = yf.Ticker(ticker_symbol)
info = company.info

# Create cards for financial metrics
col1, col2 = st.columns(2)

with col2:
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin-bottom: 15px;'>Company Financials</h3>
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>Total Revenue</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>${:,.2f}B</p>
            </div>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>Net Income</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>${:,.2f}B</p>
            </div>
        </div>
    </div>
    """.format(
        info.get('totalRevenue', 0)/1e9 if info.get('totalRevenue') else 0,
        info.get('netIncomeToCommon', 0)/1e9 if info.get('netIncomeToCommon') else 0
    ), unsafe_allow_html=True)

with col1:
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin-bottom: 15px;'>Basic Stock Info</h3>
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>Current Price</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>${:,.2f}</p>
            </div>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>Market Cap</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>${:,.2f}B</p>
            </div>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>52 Week High</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>${:,.2f}</p>
            </div>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>52 Week Low</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>${:,.2f}</p>
            </div>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>Volume</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>{:,.0f}</p>
            </div>
            <div style='background-color: #333333; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color: #888888; font-size: 14px;'>Avg Volume</p>
                <p style='margin:0; color: white; font-size: 18px; font-weight: bold;'>{:,.0f}</p>
            </div>
        </div>
    </div>
    """.format(
        info.get('currentPrice', 0),
        info.get('marketCap', 0)/1e9,
        info.get('fiftyTwoWeekHigh', 0),
        info.get('fiftyTwoWeekLow', 0),
        info.get('volume', 0),
        info.get('averageVolume', 0)
    ), unsafe_allow_html=True)

# Rest of your existing code for charts and predictions...
left, right = st.columns([10,10])
right.subheader(f"{selected_ticker} stock history")
right.dataframe(data, width=1000, height=350)



# Graph for plotting "adj close" of stock
def adj_close_graph(data):
    fig=px.line(data,x=data.index,y=data["Adj Close"].values.reshape(-1),title=f"          Adj. Close Price for {selected_ticker}")
    fig.update_layout(xaxis_title="Date",yaxis_title="Adj. Close")
    return fig
fig=adj_close_graph(data)
left.plotly_chart(fig)

# Graph for plotting "volume" of stock
# def volume_graph(data):
#     fig=px.line(data,x=data.index,y=data["Volume"].values.reshape(-1),title=f"         Stock volume for {selected_ticker}")
#     fig.update_layout(xaxis_title="Date",yaxis_title="Volume")
#     return fig
# fig=volume_graph(data)
# st.plotly_chart(fig)

data['Daily Return'] = data['Adj Close'].pct_change() * 100

# Plot volatility (daily return)
def volatility_graph(data):
    fig = px.line(data, x=data.index, y=data['Daily Return'], title=f"          Volatility (Daily Returns) for {selected_ticker}")
    fig.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)", yaxis=dict(tickformat=".2f"))
    return fig

fig = volatility_graph(data)
st.plotly_chart(fig)

#Moving Averages:
st.markdown("### **Moving Averages**")
st.write('''In finance, a moving average (MA) is a stock indicator commonly used in technical analysis. The reason for calculating the moving average of a stock is to help smoothen short-term fluctuations and highlight long-term trends in data.
 A rising moving average indicates that the security is in an uptrend, while a declining moving average indicates a downtrend.''')

def moving_average_plot(data):
    ma = [100, 200, 300]
    fig = px.line(data, x=data.index, y=data["Adj Close"].values.reshape(-1), title=f'         Adjusted Close Prices and Moving Averages for {selected_ticker}')

    for m in ma:
        rolling_avg = data["Adj Close"].rolling(m).mean()
        fig.add_scatter(x=data.index, y=rolling_avg, mode='lines', name=f'Moving average of {m} days')

    fig.update_layout(xaxis_title="Date", yaxis_title="Adj. Close", legend_title="Data Series")
    return fig

fig = moving_average_plot(data)
st.plotly_chart(fig)

# Scaling the data
# Preparing the training and testing data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data[["Adj Close"]])
X_data=[]
y_data=[]
for i in range(100,len(scaled_data)):
    X_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
splitting_length=int(len(X_data)*0.7)
# Splitting the data (70% training and 30% testing)
X_train=X_data[:splitting_length]
y_train=y_data[:splitting_length]
X_test=X_data[splitting_length:]
y_test=y_data[splitting_length:]
X_train,y_train=np.array(X_train),np.array(y_train)
X_test,y_test=np.array(X_test),np.array(y_test)
# Loading the model
model=load_model("stock_model_1.keras")
predictions=model.predict(X_test)
actual_predictions=scaler.inverse_transform(predictions)
inv_y_test=scaler.inverse_transform(y_test)
# Make the predictions
st.subheader("Original vs Predictions")
left,right=st.columns([5,4])
plotting_data=pd.DataFrame({"Original Prices":inv_y_test.reshape(-1),"Predictions":actual_predictions.reshape(-1)},index=data.index[splitting_length+100:])
left.dataframe(plotting_data,width=800)
def org_pred_graph(data):
    fig=px.line(pd.concat([data["Adj Close"][:splitting_length+100],plotting_data]))
    fig.update_layout(legend_title="")
    return fig
fig=org_pred_graph(data)
right.plotly_chart(fig)
st.divider()
# Get the last 100 days of data
last_100_days = data['Adj Close'][-100:].values.reshape(-1, 1)

# Scale the data
last_100_days_scaled = scaler.transform(last_100_days)

# Reshape the data for the model
input_data = last_100_days_scaled.reshape(1, 100, 1)

# Make the prediction
prediction_scaled = model.predict(input_data)

# Inverse transform the prediction to get the actual price
predicted_price = scaler.inverse_transform(prediction_scaled)[0][0]

st.markdown("""
<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; margin: 20px 0;'>
    <div style='background: linear-gradient(45deg, #4a4a4a, #2a2a2a); padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='color: #ffffff; margin: 0 0 10px 0; font-size: 1.5rem;'>📈 {0} Prediction</h2>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <p style='color: #cccccc; margin: 0; font-size: 1rem;'>Date: {1}</p>
                <p style='color: #cccccc; margin: 0; font-size: 1rem;'>Ticker: {2}</p>
            </div>
            <div style='text-align: right;'>
                <p style='color: #ffffff; margin: 0; font-size: 2rem; font-weight: bold;'>${3:,.2f}</p>
                <p style='color: #888888; margin: 0; font-size: 0.9rem;'>Adjusted Close Price</p>
            </div>
        </div>
    </div>
</div>
""".format(
    selected_ticker,
    datetime.now().strftime('%b %d, %Y'),
    company_map[selected_ticker],
    predicted_price
), unsafe_allow_html=True)
