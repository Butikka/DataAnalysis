#!/usr/bin/env python
# coding: utf-8

# ## Read the pickle file that has all raw aggregated stock data and save df to main() for further use.


import requests
import pandas as pd
# from datetime import datetime, timedelta, date
import datetime
import time
# from polygon import RESTClient
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


def read_and_extract_from_file(filepath):
    try:
        with open(filepath, "rb") as file:
            compressed_data = file.read()
            data = pickle.loads(lz4.frame.decompress(compressed_data))
            # print(f"Data from {filepath}: {data[:1]}")
            return data
    except FileNotFoundError:
        print(f"No file found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None



# Process files and inspect data
def process_files_in_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.pickle.lz4"))

    df_list = []

    for file in files:
        symbol = os.path.basename(file).split('-')[0]
        data = read_and_extract_from_file(file)

        if data is None or len(data) == 0:
            continue

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Only proceed if the expected columns exist
        if {'close', 'open', 'high', 'low', 'volume', 'timestamp'}.issubset(df.columns):
            df_filtered = df[['close', 'open', 'high', 'low', 'volume', 'timestamp']].copy()
            df_filtered['symbol'] = symbol
            df_filtered = df_filtered[['symbol', 'close', 'open', 'high', 'low', 'volume', 'timestamp']]
            # Convert the 'timestamp' column from milliseconds to datetime
            df_filtered['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('h')
            df_list.append(df_filtered)
        else:
            print(f"Missing expected columns in {file}. Columns available: {df.columns}")

    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df
    else:
        print("No data was processed.")
        return None