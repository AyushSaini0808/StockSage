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

company_map={"Google":"GOOG","Netflix":"NFLX","Amazon":"AMZN","Apple":"AAPL","Tesla":"TSLA","Mircosoft":"MSFT","Meta":"META"}
myKeys = list(company_map.keys())
myKeys.sort()
company_map = {i: company_map[i] for i in myKeys}
page_bg_img="""
<style>
[data-testid="stAppViewContainer"]{
    background-image: linear-gradient(to right top, #6c51d7, #6055d5, #5359d3, #475cd0, #3b5fcd, #335ac7, #2b56c2, #2251bc, #1e43b3, #1d35a9, #1e269f, #201494);
}
[data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"]{
background-image: linear-gradient(to right top, #2a1c60, #282565, #262d68, #26356c, #283c6e, #253c73, #203d78, #1b3d7d, #0d3685, #062e8c, #102491, #201494);
}
</style>
""" 
st.set_page_config(page_title="StockSage",layout="wide",page_icon="📈",initial_sidebar_state="expanded")
st.markdown(page_bg_img,unsafe_allow_html=True)
title=st.markdown('''
         # *StockSage* : **An Intelligent Stock Price Predictor**''')
st.divider()
st.sidebar.markdown("### About the Project:")
st.sidebar.info("""StockSage is designed to provide insightful information regarding the stock trends of some of the major tech. companies. Utilizing LSTM(Long-Short Term Memory),a recurrent neural network, StockSage analyzes historical stock data to forecast future price movements with high accuracy.""")

   
   

st.markdown("## Currently showing")
selected_ticker=st.selectbox("",list(company_map.keys()),placeholder="Select a ticker",)

end=datetime.now()
start=datetime(year=end.year-10,month=end.month,day=end.day)
data=yf.download(company_map[selected_ticker],start,end)

left,middle,right=st.columns([2,10,1])
middle.subheader(f"{selected_ticker} stock history")
middle.dataframe(data,width=1000,height=350)
# Graph for plotting "adj close" of stock
def adj_close_graph(data):
    fig=px.line(data,x=data.index,y=data["Adj Close"],title=f"          Adj. Close Price for {selected_ticker}")
    fig.update_layout(xaxis_title="Date",yaxis_title="Adj. Close")
    return fig
fig=adj_close_graph(data)
st.plotly_chart(fig)

# Graph for plotting "volume" of stock
def volume_graph(data):
    fig=px.line(data,x=data.index,y=data["Volume"],title=f"         Stock volume for {selected_ticker}")
    fig.update_layout(xaxis_title="Date",yaxis_title="Volume")
    return fig
fig=volume_graph(data)
st.plotly_chart(fig)
#Moving Averages:

st.markdown("### **Moving Averages**")
st.write('''In finance, a moving average (MA) is a stock indicator commonly used in technical analysis. The reason for calculating the moving average of a stock is to help smoothen short-term fluctuations and highlight long-term trends in data.
 A rising moving average indicates that the security is in an uptrend, while a declining moving average indicates a downtrend.''')

def moving_average_plot(data):
    ma = [100, 200, 300]
    fig = px.line(data, x=data.index, y=data["Adj Close"], title=f'         Adjusted Close Prices and Moving Averages for {selected_ticker}')

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

st.subheader("Original vs Predictions")
left,middle,right=st.columns([5,1,4])
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
st.markdown(f"          ##### Predicted Adusted Close price for _{selected_ticker}_ on :green[{datetime.now().date()}]: $ :blue-background[{predicted_price}]")

