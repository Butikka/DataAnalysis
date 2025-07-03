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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.parameters_linear_regression import process_all_functions
from data.read_aggregates import process_files_in_folder
from backtesting.linear_regression.filter_linear_regression_alerts import filter_data_by_symbols
from backtesting.linear_regression.entry_exit_linear_regression import entry_exit_for_all_symbols



# Path to files
path_to_files = "C:\\Users\\mustosa\\Documents\\Omat Projektit\\Investing\\data\\day\\10b_50b"

# Read readRawAggs
df = process_files_in_folder(path_to_files)

# Execute functions that create parameters for the model build and get the DataFrame
df = process_all_functions(path_to_files)

# Get the unique symbols from the DataFrame
df = filter_data_by_symbols(df)

# Calculate entry and exit positions for each symbol
df = entry_exit_for_all_symbols(df)

#Sort the data by date
# df = df.sort_values(by=['symbol', 'timestamp'])

# print(df['close'].dtypes)

# print(tabulate(df.tail(10), headers="keys", tablefmt="simple"))
df = df[df['exitPos'] == 1]
print(df.head(10))
# print(df.columns.tolist())
