import requests
import pandas as pd
# from datetime import datetime, timedelta, date
from datetime import datetime
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
import gzip
from tabulate import tabulate
from sklearn.linear_model import LinearRegression


# Calculate entry and exit positions for each symbol based on linear regression buy/sell zones.
# Logic for entry and exit.
def calculate_entry_exit_for_symbol(df_symbol):
    df_symbol = df_symbol.reset_index(drop=True)
    df_symbol['entryPos'] = 0
    df_symbol['exitPos'] = 0

    has_position = False  # Tracks if a stock is currently "owned"
    entry_counter = 1
    exit_counter = 1

    for i in range(len(df_symbol)):
        row = df_symbol.iloc[i]

        if not has_position and row['linRegBuyZone'] == 1:
            # Look ahead 1 row to assign entry flag
            if i + 1 < len(df_symbol):
                df_symbol.at[i + 1, 'entryPos'] = entry_counter
                entry_counter += 1
                has_position = True
        elif has_position and row['linRegSellZone'] == 1:
            # Look ahead 1 row to assign exit flag
            if i + 1 < len(df_symbol):
                df_symbol.at[i + 1, 'exitPos'] = exit_counter
                exit_counter += 1
                has_position = False

    return df_symbol


# Adds all calculated entry and exit positions for all symbols in the DataFrame.
def entry_exit_for_all_symbols(df):
    result_dfs = []

    for symbol in df['symbol'].unique():
        df_symbol = df[df['symbol'] == symbol].copy()
        df_symbol = calculate_entry_exit_for_symbol(df_symbol)
        result_dfs.append(df_symbol)

    return pd.concat(result_dfs, ignore_index=True)