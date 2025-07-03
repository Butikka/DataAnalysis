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
from model.parameters_linear_regression import process_all_functions



# Function to filter date range and symbols that are alerted by linear regression model.
def model_linear_regression_list(data):
    
    # Filter only rows where 'linRegBuyZone' is 1
    linRegAlert = (data['linRegBuyZone']==1)

     # Apply filters and select unique values from the 'symbol' column
    linReg_list = data[linRegAlert]['symbol'].unique()

    return linReg_list


# Filter and sort the original DataFrame using the list of unique symbols
def filter_data_by_symbols(df):

    # Run model linear regression list to create symbol list
    symbols_list = model_linear_regression_list(df)
    
    # Filter the DataFrame to include only rows where 'symbol' is in the symbols_list
    filtered_df = df[df['symbol'].isin(symbols_list)]

    # Sort the filtered DataFrame by 'symbol' and 'timestamp'
    sorted_df = filtered_df.sort_values(by=['symbol', 'timestamp'])

    return sorted_df