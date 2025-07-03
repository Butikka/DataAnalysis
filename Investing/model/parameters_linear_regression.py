#!/usr/bin/env python
# coding: utf-8

# ## Adding parameters to the data model
# **exceptionalCandleSize** = Candle size as calculated (high-low). If the candle size is 2 standard deviations larger than average candle size then 1 else 0.
# **ema20** = exponential moving average 20 days.
# **ema50** = exponential moving average 50 days.
# **openHigher** = The Gap-Up indicator: day opens 5% higher than perevious close. If Open > (1,05 * Previous day Close) then 1 else 0.
# **averageVolume** = Average volume from past 100 market-days.
# **strongVolume** = If Volume > (2 * averageVolume) then 1 else 0.
# **strongVolume6MoPrior** = If strongVolume is equal to 1 more than 3 times in the past 100 market-days (5-6 months) then 1 else 0.
# **accVolume** = If strongVolume = 1 and strongVolume6MoPrior = 1 then 1 else 0.
# **uptrend** = The stock is uptrending in the past 5 market-days. If ema50 >= (ema50 5 market-days prior) then 1 else 0.
# **closeHigh** = The close is higher than the open, thus producing a green candle.


import requests
import pandas as pd
# from datetime import datetime, timedelta, date
import datetime
import time
from polygon import RESTClient
import logging
import signal
import sys
import pickle
import lz4.frame  # type: ignore
import concurrent.futures
import os
import pandas as pd
import numpy as np
import glob
import nbimporter
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from data.read_aggregates import process_files_in_folder


# Function to calculate EMA
def calculate_ema(data, window):
    """Calculate the Exponential Moving Average (EMA) for a given window."""
    # Ensure the 'close' column is of type float
    data['close'] = data['close'].astype(float)
    # Calculate EMA using ewm() and handle insufficient data gracefully
    ema = data.groupby('symbol')['close'].ewm(span=window, adjust=False).mean()
    # Handle insufficient data gracefully
    ema = ema.reset_index(level=0, drop=True)  # Reset index to align with original DataFrame
    # Replace initial values with NaN for insufficient data
    if len(data) < window:
        ema.iloc[:window] = None  # Set initial window values to None (NaN)

    return ema


def calculate_sma(data, window):
    """Calculate the Simple Moving Average (SMA) for a given window."""
    # Ensure the 'close' column is of type float
    data['close'] = data['close'].astype(float)
    
    # Calculate SMA using rolling().mean() per symbol
    sma = data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=window, min_periods=window).mean())
    
    return sma


# Function to calculate high - low
def calculate_candle_size(data):
    """Calculate high - low to determine candle size."""
    # Ensure columns are type float
    data['high'] = data['high'].astype(float)
    data['low'] = data['low'].astype(float)
    # Calculate new column high - low
    candleSize = data['high'] - data['low']

    return candleSize


# Function to calculate average candle size
def calculate_avg_candle_size(data, window):
    """Calculate the average high-low for a given window."""
    # Ensure the column is of type float
    data['candleSize'] = data['candleSize'].astype(float)
    # Calculate average candle size and handle insufficient data gracefully
    avgCandleSize = data.groupby('symbol')['candleSize'].transform(lambda x: x.rolling(window).mean())
    # Handle insufficient data gracefully
    avgCandleSize = avgCandleSize.reset_index(level=0, drop=True)  # Reset index to align with original DataFrame
    # Replace initial values with NaN for insufficient data
    if len(data) < window:
        avgCandleSize[:window] = None  # Set initial window values to None (NaN)

    return avgCandleSize


# Function to calculate strong volume
def calculate_volatility(data):
    """Calculate the volatility for ATR."""

    # Ensure the 'volume' and 'averageVolume' columns are of type float
    data['ma14CandleSize'] = data['ma14CandleSize'].astype(float)
    data['close'] = data['close'].astype(float)

    # Calculate strong volume: volume greater than 2 times averageVolume
    volatility = (data['ma14CandleSize'] / data['close']).astype(float)

    # Replace initial values with NaN for insufficient data
    if len(data) < 14:
        volatility[:14] = None  # Set initial values to None (NaN)

    return volatility


def calculate_std_dev_candle_size(data, window):
    """Calculate the standard deviation of candle size for each symbol."""
    # Group by 'symbol' and calculate standard deviation of candleSize column
    stdDevCandleSize = data.groupby('symbol')['candleSize'].transform(lambda x: x.rolling(window).std())
    # Handle insufficient data gracefully
    stdDevCandleSize = stdDevCandleSize.reset_index(level=0, drop=True)
    if len(data) < window:
        stdDevCandleSize[:window] = None  # Set initial window values to None (NaN)

    return stdDevCandleSize


def calculate_exceptional_candle_size(data):
    """Calculate if candleSize is 2 standard deviations higher than avgCandleSize."""
    # Group by 'symbol' and calculate exceptional candle size
    exceptionalCandleSize = (data['candleSize'] >= (data['ma100CandleSize'] + 2 * data['stdDevCandleSize'])).astype(int)

    return exceptionalCandleSize


# Function to calculate average volume
def calculate_avg_volume(data, window):
    """Calculate the Average Volume for a given window."""
    # Ensure the 'volume' column is of type float
    data['volume'] = data['volume'].astype(float)
    # Calculate average volume and handle insufficient data gracefully
    averageVolume = data.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean())
    # Handle insufficient data gracefully
    averageVolume = averageVolume.reset_index(level=0, drop=True)  # Reset index to align with original DataFrame
    # Replace initial values with NaN for insufficient data
    if len(data) < window:
        averageVolume[:window] = None  # Set initial window values to None (NaN)

    return averageVolume


def calculate_ema50_downtrend(data, slope_window=3, slope_threshold=0.0):
    """
    Calculate downtrend signal based on 50-day rolling slope of ema50.

    Parameters:
    - data: DataFrame with 'symbol' and 'ema50' columns
    - slope_window: Number of days to calculate rolling slope over (default: 50)
    - slope_threshold: Minimum average slope to consider it an uptrend (default: 0.05)

    Returns:
    - DataFrame with added columns:
        - 'ema50_slope': rolling slope
        - 'ema50_uptrend': 1 if slope > threshold, else 0
    """
    data = data.copy()
    data['ema50'] = data['ema50'].fillna(0).astype(float)

    # Compute rolling slope of ema50
    data['ema50_slope'] = (
        data.groupby('symbol')['ema50']
        .transform(lambda x: x.diff().rolling(window=slope_window).mean())
    )

    # Mark downtrend if slope is less than threshold
    data['ema50_downtrend'] = (data['ema50_slope'] < slope_threshold).astype(int)

    return data


def calculate_ema20_downtrend(data, slope_window=5, slope_threshold=0.0):
    """
    Calculate downtrend signal based on 20-day rolling slope of ema20.

    Parameters:
    - data: DataFrame with 'symbol' and 'ema20' columns
    - slope_window: Number of days to calculate rolling slope over (default: 50)
    - slope_threshold: Minimum average slope to consider it an uptrend (default: 0.05)

    Returns:
    - DataFrame with added columns:
        - 'ema20_slope': rolling slope
        - 'ema20_uptrend': 1 if slope > threshold, else 0
    """
    data = data.copy()
    data['ema20'] = data['ema20'].fillna(0).astype(float)

    # Compute rolling slope of ema20
    data['ema20_slope'] = (
        data.groupby('symbol')['ema20']
        .transform(lambda x: x.diff().rolling(window=slope_window).mean())
    )

    # Mark downtrend if slope is less than threshold
    data['ema20_downtrend'] = (data['ema20_slope'] < slope_threshold).astype(int)

    return data


def calculate_macd(df):
    """
    Calculate MACD (12, 26, 9) and signal line trend for each symbol.
    Could be imported from polygon as an indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'symbol', 'close']

    Returns:
        pd.DataFrame: Original DataFrame with additional columns:
            - 'macd_line'
            - 'macd_signal'
            - 'macd_histogram'
            - 'macd_signal_trend' ('up', 'down', or NaN)
    """
    result_df_list = []

    for symbol in df['symbol'].unique():
        df_symbol = df[df['symbol'] == symbol].sort_values('timestamp').copy()
        
        # Calculate EMAs
        ema_fast = df_symbol['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df_symbol['close'].ewm(span=26, adjust=False).mean()

        # MACD and signal line
        df_symbol['macd_line'] = ema_fast - ema_slow
        df_symbol['macd_signal'] = df_symbol['macd_line'].ewm(span=9, adjust=False).mean()
        df_symbol['macd_histogram'] = df_symbol['macd_line'] - df_symbol['macd_signal']

        # Track histogram direction
        df_symbol['macd_h_trend'] = df_symbol['macd_histogram'].diff().apply(
            lambda x: 'up' if x > 0 else ('down' if x < 0 else None)
        )

        # Check if histogram was up three consecutive days
        df_symbol['macd_h_up_3d'] = (
            (df_symbol['macd_histogram'].shift(1) > 0) &
            (df_symbol['macd_histogram'].shift(2) > 0) &
            (df_symbol['macd_histogram'].shift(3) > 0)
        ).apply(lambda x: 'yes' if x else 'no')

        # macd_line higher than macd_signal (macd line higher equals uptrend)
        df_symbol['macd_s_trend'] = np.where(
            df_symbol['macd_line'] > df_symbol['macd_signal'], 'up',
            np.where(df_symbol['macd_line'] < df_symbol['macd_signal'], 'down', None)
        )

        # macd_line under 0
        df_symbol['macd_line_below_0'] = np.where(
            df_symbol['macd_line'] > 0, 'no',
            np.where(df_symbol['macd_line'] < 0, 'yes', None)
        )

        result_df_list.append(df_symbol)

    return pd.concat(result_df_list, ignore_index=True)


# --- Calculate Linear Regression Channel ---
def add_linear_regression_channel(df, window=100, std_multiplier=2):
    df['linRegMed'] = np.nan
    df['linRegHigh'] = np.nan
    df['linRegLow'] = np.nan
    
    tickers = df['symbol'].unique()
    
    for ticker in tickers:
        df_ticker = df[df['symbol'] == ticker].copy()
        closes = df_ticker['close'].values
        dates = np.arange(len(df_ticker))  # Use numeric index as x-axis
        
        # Initialize lists for temporary storage
        med_list = [np.nan] * len(closes)
        high_list = [np.nan] * len(closes)
        low_list = [np.nan] * len(closes)
        
        for i in range(window - 1, len(closes)):
            x = dates[i - window + 1: i + 1].reshape(-1, 1)
            y = closes[i - window + 1: i + 1]
            
            model = LinearRegression().fit(x, y)
            
            # Predict the value for the *current* point 'i' using the learned model
            # and the current date 'x[-1]' (which corresponds to 'i')
            y_pred_current_point = model.predict(dates[i].reshape(-1, 1))[0]
            
            # Calculate standard deviation of residuals for the current window
            y_pred_window = model.predict(x)
            std_dev = np.std(y - y_pred_window)

            # Store only the calculated values for the current point 'i'
            med_list[i] = y_pred_current_point
            high_list[i] = y_pred_current_point + std_multiplier * std_dev
            low_list[i] = y_pred_current_point - std_multiplier * std_dev

        # Reinsert into original df for the specific ticker
        df.loc[df['symbol'] == ticker, 'linRegMed'] = med_list
        df.loc[df['symbol'] == ticker, 'linRegHigh'] = high_list
        df.loc[df['symbol'] == ticker, 'linRegLow'] = low_list

    return df


def linreg_high_uptrend(data, slope_window=5, slope_threshold=0.0):
    """
    Calculate uptrend signal based on 5-day rolling slope of linear regression high.

    Parameters:
    - data: DataFrame with 'symbol' and 'linRegHigh' columns
    - slope_window: Number of days to calculate rolling slope over (default: 50)
    - slope_threshold: Minimum average slope to consider it an uptrend (default: 0.05)

    Returns:
    - DataFrame with added columns:
        - 'ema20_slope': rolling slope
        - 'linRegHigh_slope': rolling slope
        - 'linRegHigh_uptrend': 1 if slope > threshold, else 0
    """
    data = data.copy()
    data['linRegHigh'] = data['linRegHigh'].fillna(0).astype(float)

    # Compute rolling slope of linRegHigh
    data['linRegHigh_slope'] = (
        data.groupby('symbol')['linRegHigh']
        .transform(lambda x: x.diff().rolling(window=slope_window).mean())
    )

    # Mark uptrend if slope is greater than threshold
    data['linRegHigh_uptrend'] = (data['linRegHigh_slope'] > slope_threshold).astype(int)

    return data


# Function to generate buy/sell flags based on linear regression channel
def generate_buy_sell_flags(df):
    df = df.copy()
    df['linRegBuyZone'] = 0
    df['linRegSellZone'] = 0

    in_buy_zone = False
    in_sell_zone = False
    df['ema20_downtrend'] = calculate_ema20_downtrend(df)['ema20_downtrend']
    df['linRegHigh_uptrend'] = linreg_high_uptrend(df)['linRegHigh_uptrend']

    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row['close']
        low = row['linRegLow']
        mid = row['linRegMed']
        high = row['linRegHigh']
        ema50 = row['ema50']
        macd_h = row['macd_h_trend']
        macd_h_up_3d = row['macd_h_up_3d'] # 'yes' if macd histogram was up three consecutive days
        macd_line_below_0 = row['macd_line_below_0'] # 'yes' if macd line is below 0
        ema20_downtrend = row['ema20_downtrend'] # 1 if ema20 slope is downtrend
        linRegHigh_uptrend = row['linRegHigh_uptrend'] # 1 if linRegHigh slope is uptrend

        if pd.isna(low) or pd.isna(mid) or pd.isna(high) or pd.isna(ema50):
            continue

        # Sell and buy trhesholds and conditions    
        buy_threshold = low + 0.25 * (mid - low)
        sell_threshold = high - 0.25 * (high - mid) # original
        buy_condition_trends = ema20_downtrend == 1 and linRegHigh_uptrend == 0

        # Check past 30 rows (excluding current row)
        window_start = max(0, i - 30)
        past_window = df.iloc[window_start:i+1]
        window_start_sell = max(0, i-1) # past 1 row for sell condition
        past_window_sell = df.iloc[window_start_sell:i+1]
        
        # Condition: has the signal triggered at least once in past 30 days?
        past_buy_signal = (
            (past_window['close'] <= past_window['linRegLow'] + 0.25 * (past_window['linRegMed'] - past_window['linRegLow'])) 
            ).any()
        
        past_sell_signal = (
            (past_window_sell['close'] >= past_window_sell['linRegHigh'] - 0.25 * (past_window_sell['linRegHigh'] - past_window_sell['linRegMed'])) &
            (past_window_sell['macd_h_trend'] == 'down')).any()

        if not in_buy_zone and close <= buy_threshold and close < ema50:
            df.at[df.index[i], 'linRegBuyZone'] = 1
            in_buy_zone = True
        elif close > mid:
            in_buy_zone = False

        if not in_sell_zone and close >= sell_threshold and macd_h == 'down':
            df.at[df.index[i], 'linRegSellZone'] = 1
            in_sell_zone = True
        elif close < mid:
            in_sell_zone = False

    return df


# --- Calculate all functions and return the final DataFrame ---
def process_all_functions(path):
    
    # Read readRawAggs
    df = process_files_in_folder(path)

    # Sort the data by date
    df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

    # Parse 'timestamp' to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    # df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('h')

    # start_date = pd.to_datetime('2022-09-30').date()
    # # Filter dates
    # df = df[df['timestamp'] >= start_date]

    # Calculate candle size from high minus low.
    df['candleSize'] = calculate_candle_size(df)

    # Calculate ma 14 candle size
    df['ma14CandleSize'] = calculate_std_dev_candle_size(df, 14)

    # Calculate volatility for ATR (average true range as %).
    df['volatility'] = calculate_volatility(df)

    # Check only one symbol for testing purposes.
    # df = df[df['symbol'] == 'AAOI']
    # Get unique symbols and select the first 50 or list of symbols to analyze.
    # df = df[df['symbol'].isin(['AMCR', 'AFG', 'AKAM', 'ABEV'])].copy()
    first_50_symbols = df['symbol'].unique()[:50]
    # Filter the DataFrame for first 50 symbols
    df = df[df['symbol'].isin(first_50_symbols)].copy()

    # Create columns for EMA20 and EMA50.
    df['ema20'] = calculate_ema(df, 20)
    df['ema50'] = calculate_ema(df, 50)

    # Create columns for SMA10 and SMA20.
    df['sma10'] = calculate_sma(df, 10)
    df['sma20'] = calculate_sma(df, 20)

    # MACD calculation
    df = calculate_macd(df)

    # df with linear regression channel and buy and sell flags.
    df = add_linear_regression_channel(df)
    df = generate_buy_sell_flags(df)
    
    # Round columns to 2 decimal places and one to 4.
    df[['close', 'open', 'high', 'low', 'ema50', 'linRegMed', 'linRegHigh', 'linRegLow']] = df[['close', 'open', 'high', 'low', 'ema50', 'linRegMed', 'linRegHigh', 'linRegLow']].round(2)

    # Round the large value column to the nearest hundred without decimals
    df['volume'] = (df['volume'] / 100).round() * 100
    df['volume'] = (df['volume']).astype(int)

    return df


# ---- Additional calculations and columns to use ----

# # Calculate candle size from high minus low.
# df['candleSize'] = calculate_candle_size(df)

# # Calculate std dev candle size from high minus low.
# df['stdDevCandleSize'] = calculate_std_dev_candle_size(df, 100)

# # Calculate ma 14 candle size
# df['ma14CandleSize'] = calculate_std_dev_candle_size(df, 14)

# # Candle standard deviation in percentage
# df['stdDevCandleSizePer'] = (df['stdDevCandleSize'] / df['candleSize'])

# # Trailing stop price is 1 - (( 1 + std% ) * volatility) * high. MUUTOS: OPEN TULEE OLLA KORKEAMMALLA KUIN CLOSE (PUNAINEN KYNTTILÄ)
# df['trailingStop'] = (1 - (( 1 + df['stdDevCandleSizePer'] ) * df['volatility'])) * df['high']

# # Calculate average volume from past 100 days.
# df['averageVolume'] = calculate_avg_volume(df, 100)


# Check how the data looks like.
# print(tabulate(df.tail(20), headers="keys", tablefmt="simple"))

# # Drop unecessary columns from dataframe.
# df.drop(['candleSize', 'ma100CandleSize', 'prev_close', 'ma14CandleSize', 'stdDevCandleSize', 'stdDevCandleSizePer'], axis=1, inplace=True)


# --- GRAPH FROM MATPLOTLIB ---

# Plotting
# Plot the base price chart
# plt.plot(df['timestamp'], df['close'], label='Close', color='black')
# plt.plot(df['timestamp'], df['linRegMed'], label='Med', color='red', linestyle='--', linewidth=0.8)
# plt.plot(df['timestamp'], df['linRegHigh'], color='blue', linewidth=1.2)
# plt.plot(df['timestamp'], df['linRegLow'], color='blue', linewidth=1.2)
# plt.plot(df['timestamp'], df['ema50'], color='purple', linewidth=1.1)
# plt.plot(df['timestamp'], df['ema20'], color='green', linewidth=1.1)

# # Buy markers (green upward triangle)
# plt.scatter(
#     buy_signals['timestamp'],
#     buy_signals['close'],
#     marker='^',
#     color='green',
#     label='Buy Signal',
#     s=70
# )

# # Sell markers (red downward triangle)
# plt.scatter(
#     sell_signals['timestamp'],
#     sell_signals['close'],
#     marker='v',
#     color='red',
#     label='Sell Signal',
#     s=70
# )

# # Add text labels for close prices
# for _, row in buy_signals.iterrows():
#     plt.text(row['timestamp'], row['close'], f"{row['close']:.4f}", color='green', fontsize=8, ha='left', va='bottom')

# for _, row in sell_signals.iterrows():
#     plt.text(row['timestamp'], row['close'], f"{row['close']:.4f}", color='red', fontsize=8, ha='left', va='top')

# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.title('Linear Regression Channel with Buy/Sell Signals')
# plt.xlabel('date')
# plt.ylabel('Price')
# plt.tight_layout()
# plt.show()