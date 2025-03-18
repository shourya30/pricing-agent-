
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Generate synthetic historical price data for Glycerine (2015-2025)
np.random.seed(42)
date_range = pd.date_range(start="2015-01-01", end="2025-03-01", freq="M")

# Base price trend (gradually increasing with fluctuations)
base_price = np.linspace(800, 1500, len(date_range))

# Random fluctuations to simulate market dynamics
seasonality = 50 * np.sin(np.linspace(0, 20, len(date_range)))  # Seasonal pattern
random_variation = np.random.normal(scale=50, size=len(date_range))  # Market volatility

# Final synthetic price series
glycerine_prices = base_price + seasonality + random_variation

# Create DataFrame
glycerine_data = pd.DataFrame({
    "Date": date_range,
    "Glycerine_Price_USD_per_Ton": glycerine_prices
})

# Web App Title
st.title("🛢️ Glycerine Pricing AI Agent")

# Sidebar for User Interaction
st.sidebar.header("Select Analysis")

# Price Trend Visualization
if st.sidebar.checkbox("Show Historical Price Trend", True):
    st.subheader("📊 Historical Glycerine Price Trend")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(glycerine_data["Date"], glycerine_data["Glycerine_Price_USD_per_Ton"], label="Glycerine Price", color="blue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (USD per Ton)")
    ax.set_title("Historical Glycerine Prices (2015-2025)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Forecasting Module
if st.sidebar.checkbox("Forecast Future Prices", True):
    st.subheader("🔮 Glycerine Price Forecast (Next 12 Months)")
    
    # Fit ARIMA model
    train_size = int(len(glycerine_data) * 0.8)
    train = glycerine_data[:train_size]
    
    model = ARIMA(train["Glycerine_Price_USD_per_Ton"], order=(3,1,3))
    model_fit = model.fit()

    # Forecast future prices
    forecast_steps = 12
    future_forecast = model_fit.forecast(steps=forecast_steps)
    
    # Create future dates
    future_dates = pd.date_range(start=glycerine_data["Date"].iloc[-1], periods=forecast_steps+1, freq="M")[1:]
    
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Price": future_forecast.values})

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(glycerine_data["Date"], glycerine_data["Glycerine_Price_USD_per_Ton"], label="Historical Prices", color="blue")
    ax.plot(forecast_df["Date"], forecast_df["Forecasted_Price"], label="Forecasted Prices", color="red", linestyle="dashed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (USD per Ton)")
    ax.set_title("Glycerine Price Forecast (Next 12 Months)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Display forecasted data
    st.dataframe(forecast_df)

# AI-Based Decision Making
if st.sidebar.checkbox("AI Recommendations", True):
    st.subheader("🤖 AI-Based Market Recommendations")

    latest_price = glycerine_data["Glycerine_Price_USD_per_Ton"].iloc[-1]
    forecasted_price = forecast_df["Forecasted_Price"].iloc[0]

    if forecasted_price > latest_price:
        st.success("📈 AI Suggestion: Prices are expected to **increase**. Consider **buying now**.")
    else:
        st.warning("📉 AI Suggestion: Prices are expected to **decrease**. Consider **waiting** before buying.")

# Alerts for Significant Price Changes
if st.sidebar.checkbox("Alerts & Reporting", True):
    st.subheader("🚨 Price Change Alerts")

    # Define thresholds for alerts
    price_change_threshold = 100

    # Identify significant changes
    glycerine_data["Price_Change"] = glycerine_data["Glycerine_Price_USD_per_Ton"].diff()

    # Generate alerts
    alerts = glycerine_data[abs(glycerine_data["Price_Change"]) > price_change_threshold][["Date", "Glycerine_Price_USD_per_Ton", "Price_Change"]]

    if not alerts.empty:
        st.error("⚠️ Significant Price Changes Detected!")
        st.dataframe(alerts)
    else:
        st.success("✅ No major price fluctuations detected.")
