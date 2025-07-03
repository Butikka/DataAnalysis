import os
import logging
import concurrent.futures
from functools import wraps
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import csv
from polygon import RESTClient
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv


 
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper for Retries (as provided previously) ---
def retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
    """
    Retry calling the decorated function using an exponential backoff.
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

# --- fetch_tickers ---

# Apply retry decorator to key API calls
@retry(exceptions=(Exception,), tries=5, delay=5, backoff=2, logger=logging)
def list_tickers_with_retries(client, **kwargs):
    """Wrapper for client.list_tickers with retry logic."""
    return client.list_tickers(**kwargs)

@retry(exceptions=(Exception,), tries=5, delay=5, backoff=2, logger=logging)
def get_ticker_details_with_retries(client, ticker_symbol):
    """Wrapper for client.get_ticker_details with retry logic."""
    return client.get_ticker_details(ticker_symbol)


def fetch_tickers(client, max_workers_details_fetch: int = 5) -> pd.DataFrame:
    """
    Fetches active stock tickers and their market capitalization using Polygon.io API.
    Optimized for fewer API calls and parallel fetching of details.

    Args:
        client: An initialized Polygon RESTClient instance.
        max_workers_details_fetch (int): Number of concurrent workers for fetching ticker details.
                                         Adjust based on your Polygon.io rate limits.

    Returns:
        pd.DataFrame: A DataFrame with 'Ticker' and 'Market-Cap' columns.
    """
    logging.info("Starting to fetch active stock tickers...")
    max_tickers_to_fetch = 500 
    all_tickers = []
    try:
        # Fetch the list of tickers with pagination.
        # The list_tickers endpoint is paginated, so iterate through it.
        # We use a broad try-except for the entire list_tickers operation,
        # relying on the decorator for individual page retries if it supports.
        # If client.list_tickers itself doesn't internally handle pagination and retries
        # robustly, you might need to add a loop with explicit pagination.
        # The Polygon client's list_tickers method typically handles pagination internally.
        for t in list_tickers_with_retries(client, market="stocks", active=True, limit=1000):
            # Only append if the ticker symbol exists
            if hasattr(t, 'ticker') and t.ticker:
                all_tickers.append(t.ticker)
                if len(all_tickers) >= max_tickers_to_fetch:
                    logging.info(f"Reached max_tickers_to_fetch limit of {max_tickers_to_fetch}.")
                    break # Stop fetching more tickers
        logging.info(f"Found {len(all_tickers)} active stock tickers.")

    except Exception as e:
        logging.error(f"Failed to list tickers: {e}")
        return pd.DataFrame(columns=['Ticker', 'Market-Cap']) # Return empty DF on failure

    # Fetch details for each ticker in parallel
    ticker_data_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_details_fetch) as executor:
        # Submit tasks for fetching details
        future_to_ticker = {executor.submit(get_ticker_details_with_retries, client, ticker_symbol): ticker_symbol for ticker_symbol in all_tickers}

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_symbol = future_to_ticker[future]
            try:
                details = future.result()
                market_cap = getattr(details, 'market_cap', None)
                ticker_data_list.append({'Ticker': ticker_symbol, 'Market-Cap': market_cap})
            except Exception as e:
                logging.warning(f"Skipping details for {ticker_symbol} due to error: {e}")

    logging.info(f"Successfully fetched details for {len(ticker_data_list)} tickers.")

    # Convert the list of dicts to a DataFrame
    df = pd.DataFrame(ticker_data_list)
    return df

# --- filter_and_sort_tickers ---

def filter_and_sort_tickers(df: pd.DataFrame,
                            min_market_cap: int = 10_000_000_000,  # 10 billion USD
                            max_market_cap: int = 50_000_000_000,  # 50 billion USD
                            limit_rows: int = 500) -> pd.DataFrame:
    """
    Filters and sorts a DataFrame of tickers based on market capitalization.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Ticker' and 'Market-Cap' columns.
        min_market_cap (int): Minimum market capitalization in USD.
        max_market_cap (int): Maximum market capitalization in USD.
        limit_rows (int): Number of rows to return from the end of the sorted DataFrame.

    Returns:
        pd.DataFrame: Filtered and sorted DataFrame.
    """
    logging.info(f"Filtering tickers with market cap between ${min_market_cap:,.0f} and ${max_market_cap:,.0f}.")

    if df.empty:
        logging.warning("Input DataFrame is empty, returning empty DataFrame.")
        return pd.DataFrame(columns=['Ticker', 'Market-Cap'])

    # Fill NaN 'Market-Cap' values with 0 before conversion to int.
    # This handles cases where market_cap might be missing from API response.
    # It's better to explicitly handle NaNs rather than implicitly converting them to large integers.
    df['Market-Cap'] = df['Market-Cap'].fillna(0).astype(float) # Ensure float for comparison
    
    # Filter the DataFrame based on the market cap range
    filtered_df = df[
        (df['Market-Cap'] >= min_market_cap) &
        (df['Market-Cap'] <= max_market_cap)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    logging.info(f"Found {len(filtered_df)} tickers within the specified market cap range.")

    # Sort by Market-Cap in ascending order
    filtered_df = filtered_df.sort_values(by='Market-Cap', ascending=True)

    # Show the last `limit_rows` of the filtered DataFrame
    # This effectively gets the top `limit_rows` largest market caps within the filtered range
    # because it was sorted ascending and then we take the tail.
    # If you want the smallest 500, you'd use .head(500) after ascending sort.
    # If you want the largest 500, you'd use .head(500) after descending sort.
    # Given your original code's `tail(500)` after `ascending=True`, it selects the
    # 500 largest market caps within the filtered range.
    if len(filtered_df) > limit_rows:
        filtered_df = filtered_df.tail(limit_rows)
        logging.info(f"Limited results to the top {limit_rows} tickers by market cap within the range.")
    else:
        logging.info(f"Number of filtered tickers ({len(filtered_df)}) is less than or equal to the limit ({limit_rows}).")


    # Convert 'Market-Cap' to int after filtering and sorting if desired,
    # as float is better for initial comparisons and NaNs.
    filtered_df['Market-Cap'] = filtered_df['Market-Cap'].astype(int)

    return filtered_df
