# ============================================================
# StocksGrow – One Page App with Dashboard, Analysis, Forecast
# LSTM-based Stock Price Prediction
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import mysql.connector
import hashlib

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="StocksGrow | LSTM Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.block-container {padding-top:1.2rem;}
h1,h2,h3 {font-weight:800;}
.small {color:#aaa;font-size:14px;}
.stButton button {
    width:100%;
    border-radius:8px;
    background:#4c8bf5;
    color:white;
    font-weight:600;
}
.card {
    background:#111;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PASSWORD HASH (DB SAFE)
# ============================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()[:8]

# ============================================================
# MYSQL CONNECTION
# ============================================================
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Parth@1811",
        database="spps",
        auth_plugin="mysql_native_password"
    )

def check_login(email, password):
    pwd = hash_password(password)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT login_id FROM login WHERE login_email=%s AND login_password=%s",
        (email, pwd)
    )
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result is not None

def email_exists(email):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT reg_id FROM register WHERE reg_email=%s", (email,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result is not None

def register_user(name, email, password):
    pwd = hash_password(password)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO register (reg_email, password, reg_name) VALUES (%s,%s,%s)",
        (email, pwd, name)
    )
    cur.execute(
        "INSERT INTO login (login_email, login_password) VALUES (%s,%s)",
        (email, pwd)
    )
    conn.commit()
    cur.close()
    conn.close()

# ============================================================
# AUTH SCREEN
# ============================================================
def auth_screen():
    st.markdown("<h1>🔐 StocksGrow Login</h1>", unsafe_allow_html=True)
    st.markdown("<p class='small'>Secure access to AI-driven stock analytics</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            if check_login(email, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        name = st.text_input("Full Name", key="reg_name")
        reg_email = st.text_input("Email ID", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register", key="register_btn"):
            if email_exists(reg_email):
                st.error("Email already exists")
            else:
                register_user(name, reg_email, reg_password)
                st.success("Registration successful")

# ============================================================
# DATA LOADER
# ============================================================
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

# ============================================================
# TICKERS
# ============================================================
TICKERS = {
    "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
    "US Stocks": ["AAPL", "MSFT", "GOOGL", "TSLA"],
    "Oil": ["CL=F", "BZ=F"],
    "Crypto": ["BTC-USD", "ETH-USD"]
}

# ============================================================
# LSTM MODEL (CACHED)
# ============================================================
@st.cache_resource
def train_lstm(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    window = 60
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    return model, scaler, window

# ============================================================
# MAIN APP
# ============================================================
def stocks_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Stock Analysis", "Forecast"])

    asset_class = st.sidebar.selectbox("Asset Type", list(TICKERS.keys()))
    ticker = st.sidebar.selectbox("Ticker", TICKERS[asset_class])

    start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", date.today())

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("No data available")
        return

    close = data["Close"]

    # ================= DASHBOARD =================
    if page == "Dashboard":
        st.markdown("<h1>📊 Dashboard</h1>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Price", round(close.iloc[-1], 2))
        c2.metric("52W High", round(data["High"].max(), 2))
        c3.metric("52W Low", round(data["Low"].min(), 2))

        st.plotly_chart(
            px.line(data, x="Date", y="Close", title="Closing Price Trend"),
            use_container_width=True
        )

    # ================= STOCK ANALYSIS =================
    elif page == "Stock Analysis":
        st.markdown("<h1>📈 Stock Analysis</h1>", unsafe_allow_html=True)

        candle = go.Figure(data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"]
            )
        ])
        st.plotly_chart(candle, use_container_width=True)

        data["Returns"] = data["Close"].pct_change()
        st.write("Average Daily Return:", round(data["Returns"].mean()*100, 3), "%")
        st.write("Volatility:", round(data["Returns"].std()*100, 3), "%")

        info = yf.Ticker(ticker).info
        st.subheader("Stock Information")
        st.write("Sector:", info.get("sector"))
        st.write("Exchange:", info.get("exchange"))
        st.write("Currency:", info.get("currency"))

    # ================= FORECAST =================
    elif page == "Forecast":
        st.markdown("<h1>🔮 LSTM Forecast</h1>", unsafe_allow_html=True)

        horizon = st.slider("Forecast Days", 1, 30, 10)

        model, scaler, window = train_lstm(close.values)

        last_seq = close.values[-window:]
        scaled_last = scaler.transform(last_seq.reshape(-1, 1))
        X_input = scaled_last.reshape(1, window, 1)

        predictions = []
        for _ in range(horizon):
            pred = model.predict(X_input, verbose=0)[0][0]
            predictions.append(pred)
            X_input = np.roll(X_input, -1)
            X_input[0, -1, 0] = pred

        forecast_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        future_dates = pd.date_range(
            data["Date"].iloc[-1], periods=horizon+1, freq="D"
        )[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=close, name="Actual"))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_prices, name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast Price": forecast_prices
        })
        st.dataframe(forecast_df)

        st.download_button(
            "Download Forecast",
            forecast_df.to_csv(index=False).encode(),
            "lstm_forecast.csv",
            "text/csv"
        )

# ============================================================
# ENTRY POINT
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    stocks_app()
else:
    auth_screen()
