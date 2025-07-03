import requests
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple
from functools import wraps
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
from tickers import fetch_tickers, filter_and_sort_tickers
from dotenv import load_dotenv


# Load .env file into environment
load_dotenv()
# Retrieve the API key
env_key = "POLYGON_API_KEY"
api_key = os.getenv(env_key)
# Define the API details
client = RESTClient(api_key=api_key)


# Define the path where the files will be saved. Up here for convenience
save_path = "C:\\Users\\mustosa\\Documents\\Omat Projektit\\Investing\\data\\day\\10b_50b" # example saving path on device.

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)


# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


# --- Decorator for Retries ---
def retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
    """
    Retry calling the decorated function using an exponential backoff.

    Args:
        exceptions (Tuple): The exception(s) to catch. Can be a single exception or a tuple of exceptions.
        tries (int): Number of times to try (not including the first attempt).
        delay (int): Initial delay between retries in seconds.
        backoff (int): Multiplier applied to the delay between retries.
        logger (logging.Logger): Logger to use for messages.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs) # Last try
        return f_retry
    return deco_retry

# --- API Best Practices ---

# Apply the retry decorator to the client.list_aggs call
# We'll assume the Polygon RESTClient might raise HTTPStatusError for network issues or rate limits.
# We might need to import requests.exceptions.HTTPStatusError or similar, depending on the client's internal exceptions.
# For simplicity, we'll use a broad 'Exception'.
@retry(exceptions=(Exception,), tries=5, delay=5, backoff=2, logger=logging)
def list_aggs_with_retries(client, ticker, multiplier, timespan, from_, to, sort, limit):
    """
    Wrapper around client.list_aggs to apply retry logic.
    """
    return client.list_aggs(ticker, multiplier, timespan, from_, to, sort=sort, limit=limit)


# Function to retrieve and save aggregate data for a given ticker over a date range
def get_aggs_for_ticker(ticker, start_date, end_date, client):
    """Retrieve aggregates for a given ticker and date range"""
    aggs = []
    # Set a reasonable timeout for the API call (e.g., 30 seconds).
    # The Polygon client might have a default timeout, but explicitly setting it
    # or ensuring your wrapped function handles it is good practice.
    # or to specific methods if they support it.
    # in the example, we'll assume the retry mechanism will help with network flakiness.
    # If the client itself supports a timeout parameter, you'd add it to list_aggs_with_retries.

    for day in weekdays_between(start_date, end_date):
        try:
            # Use the retried version of list_aggs
            for a in list_aggs_with_retries(client, ticker, 1, "day", day, day, sort='asc', limit=5000):
                aggs.append(a)
        except Exception as e:
            # Catching specific exceptions from the API call if they are not handled by the retry decorator
            logging.error(f"Failed to retrieve aggs for {ticker} on {day}: {e}")
            # Optionally, decide if you want to continue or break for this ticker
            continue # Continue to the next day even if one day fails

    # Save the data to a compressed .pickle.lz4 file. We want it compressed as there might be a lot of data.
    filename = os.path.join(save_path, f"{ticker}-aggs-{start_date}_to_{end_date}.pickle.lz4")
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(filename, "wb") as file:
            compressed_data = lz4.frame.compress(pickle.dumps(aggs))
            file.write(compressed_data)
        logging.info(f"Downloaded aggs for {ticker} from {start_date} to {end_date} and saved to {filename}")
    except TypeError as e:
        logging.error(f"Serialization Error for {ticker}: {e}")
    except IOError as e:
        logging.error(f"File I/O Error for {ticker} at {filename}: {e}")


# Function to generate all weekdays between two dates
def weekdays_between(start_date, end_date):
    """Generate all weekdays between start_date and end_date"""
    day = start_date
    while day <= end_date:
        if day.weekday() < 5:  # Only Monday to Friday
            yield day
        day += timedelta(days=1)


def execute_aggs_fetch(df: pd.DataFrame, client):
    start_date = datetime.date(2022, 9, 1)  # Start date: From September of 2022
    end_date = datetime.date(2025, 6, 30)  # End date: 2025-06-30

    # Use tickers from df (assumes the df has columns 'Ticker' and 'Market-Cap')
    df_tickers = df

    # Extract tickers from the 'Ticker' column
    symbols = df_tickers['Ticker'].tolist()

    # Use ThreadPoolExecutor to download data for each ticker in parallel.
    # Consider rate limits of Polygon.io when setting max_workers.
    # If your subscription tier has a low rate limit, 10 might be too high.
    # Adjust max_workers based on your Polygon.io subscription tier's rate limits.
    # For example, 5 is a safer starting point.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_aggs_for_ticker, symbol, start_date, end_date, client) for symbol in symbols]

        # Optional: to make sure each task completes
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions that occurred during execution
            except Exception as e:
                logging.error(f"Error fetching data for a ticker: {e}")


# Main execution
logging.info("Starting ticker fetching and filtering process...")
df_tickers = fetch_tickers(client, max_workers_details_fetch=10) # Adjust concurrency here
df = filter_and_sort_tickers(df_tickers)
logging.info(f"Filtered DataFrame head:\n{df.head()}")
logging.info(f"Filtered DataFrame tail:\n{df.tail()}")
logging.info(f"Number of filtered tickers: {len(df)}")
logging.info("Ticker fetching and filtering process completed.")
# Execute the aggregation fetch for the filtered tickers
logging.info("Starting to fetch aggregate data for tickers...")
execute_aggs_fetch(df, client)
logging.info("Aggregate data fetching process completed.")