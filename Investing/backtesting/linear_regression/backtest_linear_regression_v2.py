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

# Execute functions that create parameters for the model build and get the DataFrame
df = process_all_functions(path_to_files)

# Get the unique symbols from the DataFrame
df = filter_data_by_symbols(df)

# Calculate entry and exit positions for each symbol
df = entry_exit_for_all_symbols(df)


# --- Main backtesting function V2 ---
def backtest(df, initial_cash, trade_fee_percent, max_open_positions, stop_loss_factor=7, first_win_factor=3, risk=0.05):
    # Sort the DataFrame by timestamp to ensure chronological processing
    df = df.sort_values(by='timestamp').copy()

    # Portfolio-wide variables
    portfolio_cash = initial_cash
    open_positions = {}  # {symbol: {'shares': int, 'buy_price': float, 'buy_date': datetime, 'stop_loss_price': float, 'take_profit_price': float}}

    # Metrics
    total_trades = 0
    total_profit = 0
    total_fees = 0
    win_trades = 0
    max_profit = 0  # Max profit from a single trade
    max_drawdown = 0
    peak_portfolio_value = initial_cash
    trades = []  # List to store individual trade details

    for index, row in df.iterrows():
        close_price = row['close']
        entryPos = row['entryPos']
        exitPos = row['exitPos']
        ticker = row['symbol']
        current_date = row['timestamp']
        volatility = row['volatility']

        # --- Sell Condition Check ---
        # Check if we hold a position in the current ticker OR if any held position meets an exit criteria
        # We need to iterate through open_positions to check for any sell conditions for held stocks
        symbols_to_sell = []
        for held_symbol, pos_info in open_positions.items():
            # Get current price for the held symbol (assuming 'row' only gives one ticker at a time)
            # This is a critical point: The 'row' only provides data for one ticker.
            # To check sell conditions for *all* open positions, you'd need to access their current prices.
            # For simplicity in this backtest, we'll only check the 'exitPos' for the 'ticker' in the current 'row'.
            # A more robust solution would involve a multi-index DataFrame or a lookup function
            # to get current prices for all held symbols at each timestamp.
            # For now, let's assume 'exitPos' and 'close_price' in the current row apply to the 'ticker' in the row.

            # If the current row's ticker is one we hold, check its specific exit condition
            if held_symbol == ticker:
                # Calculate current stop loss and take profit based on volatility at buy time
                # Note: volatility for SL/TP should ideally be based on volatility *at the time of buying*
                # For consistency we'll use volatility at buy.
                current_stop_loss = pos_info['buy_price'] * (1 - stop_loss_factor * pos_info['volatility_at_buy'])
                current_take_profit = pos_info['buy_price'] * (1 + first_win_factor * pos_info['volatility_at_buy'])

                if exitPos >= 1 or close_price <= current_stop_loss:
                    symbols_to_sell.append(held_symbol)
            
            # For this example we'll  only apply exitPos for the current row's ticker,
            # but SL/TP will be checked for all held stocks against their specific prices if available.
            # The current setup assumes 'close_price' is the relevant price for 'ticker' on 'current_date'.
            # A more robust system would involve checking `df.loc[(df['timestamp'] == current_date) & (df['symbol'] == held_symbol), 'close'].iloc[0]`
            # for each `held_symbol`'s current price.

        for sell_symbol in symbols_to_sell:
            if sell_symbol in open_positions:
                pos_info = open_positions[sell_symbol]
                shares_sold = pos_info['shares']
                buy_price_of_position = pos_info['buy_price']
                
                # Retrieve the latest close price for the sell_symbol
                # This assumes the DataFrame `df` has all relevant prices.
                # If the 'close_price' in the current row is *not* for the sell_symbol, you need to find it.
                # For simplicity, if sell_symbol matches the row's ticker, use row's close_price.
                
                # Let's assume for selling, we use the close price of the *current* row if it matches the symbol,
                # or the last known close price if it doesn't match the current row's symbol.
                # A proper backtester would fetch the exact price for 'sell_symbol' at 'current_date'.
                # For now, let's just use the 'close_price' of the current row if the symbol matches.
                
                actual_sell_price = row['close'] if ticker == sell_symbol else buy_price_of_position # Placeholder - this needs real price

                total_value = shares_sold * actual_sell_price
                fee = total_value * trade_fee_percent
                total_value -= fee
                total_fees += fee
                profit = total_value - (shares_sold * buy_price_of_position) - (shares_sold * buy_price_of_position * trade_fee_percent)
                portfolio_cash += total_value

                if profit > 0:
                    win_trades += 1

                total_profit += profit
                total_trades += 1

                trades.append({
                    'date': current_date,
                    'symbol': sell_symbol,
                    'type': 'sell',
                    'price': actual_sell_price,
                    'shares': shares_sold,
                    'fee': fee,
                    'profit': profit,
                    'buy_date': pos_info['buy_date'],
                    'buy_price': pos_info['buy_price'],
                })

                del open_positions[sell_symbol] # Remove position from open positions

        # --- Buy Condition Check ---
        # Only buy if there's an entry signal, enough cash, and we're not at max open positions
        if entryPos >= 1 and len(open_positions) < max_open_positions:
            # Check if we already hold this specific ticker, if so, skip (no averaging down/up for now)
            if ticker not in open_positions:
                cash_per_position = portfolio_cash / (max_open_positions - len(open_positions)) if (max_open_positions - len(open_positions)) > 0 else 0
                
                if cash_per_position > 0: # Ensure we have cash available for this new position
                    position_size = min(cash_per_position, portfolio_cash) # Don't overspend available cash
                    
                    if close_price > 0: # Avoid division by zero
                        shares_to_buy = int(position_size / close_price)
                    else:
                        shares_to_buy = 0

                    if shares_to_buy > 0:
                        total_cost = shares_to_buy * close_price
                        fee = total_cost * trade_fee_percent
                        total_cost += fee

                        if portfolio_cash >= total_cost: # Final check for sufficient cash after fee
                            portfolio_cash -= total_cost
                            total_fees += fee

                            # Calculate stop loss and take profit for this specific trade
                            stop_loss_price = close_price * (1 - stop_loss_factor * volatility)
                            # take_profit_price = close_price * (1 + first_win_factor * volatility)

                            open_positions[ticker] = {
                                'shares': shares_to_buy,
                                'buy_price': close_price,
                                'buy_date': current_date,
                                'stop_loss_price': stop_loss_price,
                                # 'take_profit_price': take_profit_price,
                                'volatility_at_buy': volatility # Store volatility at time of buy
                            }

                            trades.append({
                                'date': current_date,
                                'symbol': ticker,
                                'type': 'buy',
                                'price': close_price,
                                'shares': shares_to_buy,
                                'fee': fee,
                                'stop_loss_price': round(stop_loss_price, 2),
                                # 'take_profit_price': round(take_profit_price, 2),
                                'volatility': volatility
                            })

        # --- Update Portfolio Value and Drawdown ---
        current_portfolio_value = portfolio_cash
        for symbol, pos_info in open_positions.items():
            # We need to get the current price for each held symbol to accurately calculate total value
            # This is a simplification: it uses the 'close_price' of the *current row's ticker*.
            # A real backtester would need a way to look up the current price for *all* `open_positions` symbols.
            # For demonstration, let's assume 'close_price' here is for the 'ticker' of the current row.
            # If the held symbol is not the current row's ticker, its value won't update until its row appears.
            # For a more accurate portfolio value, we'd need a multi-indexed dataframe or a dictionary lookup
            # of all symbols' prices for the current timestamp.
            
            # To simulate, we'll try to get the most recent price for the held symbol from the df up to current_date
            latest_price_for_held_symbol = df[(df['symbol'] == symbol) & (df['timestamp'] <= current_date)]['close'].iloc[-1] if not df[(df['symbol'] == symbol) & (df['timestamp'] <= current_date)].empty else pos_info['buy_price']

            current_portfolio_value += pos_info['shares'] * latest_price_for_held_symbol

        if current_portfolio_value > peak_portfolio_value:
            peak_portfolio_value = current_portfolio_value
        
        # Avoid division by zero for initial periods
        if peak_portfolio_value > 0:
            drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value
            max_drawdown = max(max_drawdown, drawdown)

    # --- Final Calculations ---
    # Convert remaining open positions to cash at their last known prices
    final_portfolio_value = portfolio_cash
    for symbol, pos_info in open_positions.items():
        # Get the last known close price for the symbol from the entire DataFrame
        last_known_price = df[df['symbol'] == symbol]['close'].iloc[-1] if not df[df['symbol'] == symbol].empty else pos_info['buy_price']
        final_portfolio_value += pos_info['shares'] * last_known_price

    total_return = round((final_portfolio_value - initial_cash) / initial_cash * 100, 1)
    win_rate = round(win_trades / total_trades * 100, 1) if total_trades > 0 else 0

    # Calculate best and worst trade based on 'profit' stored in 'trades' list
    trade_profits = [t['profit'] for t in trades if t['type'] == 'sell']
    best_trade = round(max(trade_profits), 2) if trade_profits else 0
    worst_trade = round(min(trade_profits), 2) if trade_profits else 0
    
    # Calculate average profit/loss from trades
    profitable_trades = [t['profit'] for t in trades if t['type'] == 'sell' and t['profit'] > 0]
    losing_trades = [t['profit'] for t in trades if t['type'] == 'sell' and t['profit'] < 0]
    avg_win = round(sum(profitable_trades) / len(profitable_trades), 2) if profitable_trades else 0
    avg_loss = round(sum(losing_trades) / len(losing_trades), 2) if losing_trades else 0
    
    win_factor = avg_win / abs(avg_loss) if avg_loss != 0 else (avg_win / 0.0001 if avg_win > 0 else 0) # Avoid division by zero

    bad_perf = win_rate < 50

    return {
        'trades': trades,
        'starting_cash': initial_cash,
        'final_portfolio_value': round(final_portfolio_value, 2),
        'total_profit_%': total_return,
        'total_profit': round(total_profit, 2),
        'total_fees': round(total_fees, 2),
        'total_trades': total_trades,
        'win_rate': win_rate,
        'best_trade_profit': best_trade,
        'worst_trade_profit': worst_trade,
        'avg_win_profit': avg_win,
        'avg_loss_profit': avg_loss,
        'win_factor': round(win_factor, 2),
        'max_drawdown': round(max_drawdown * 100, 2),  # in %
        'bad_perf': bad_perf,
    }

# --- Execute Backtesting V2 ---

# Example DataFrame structure (for testing)
# data = {
#     'timestamp': pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-02', '2025-01-03', '2025-01-03', '2025-01-04', '2025-01-04', '2025-01-05', '2025-01-05']),
#     'symbol': ['ABC', 'DEF', 'ABC', 'DEF', 'ABC', 'DEF', 'ABC', 'DEF', 'ABC', 'DEF'],
#     'close': [100, 50, 101, 51, 98, 52, 105, 49, 102, 55],
#     'entryPos': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0], # Buy signal for ABC on 01-01, DEF on 01-02, ABC on 01-04
#     'exitPos': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # Sell signal for ABC on 01-02, DEF on 01-03, DEF on 01-04
#     'volatility': [0.01, 0.02, 0.015, 0.025, 0.018, 0.022, 0.012, 0.028, 0.016, 0.021]
# }
# df = pd.DataFrame(data)


# Ensure timestamp is datetime and sort by timestamp then symbol for consistent order
# df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('h')
df = df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)

# Run the backtest once for the entire portfolio
# Set your desired initial cash and max open positions
initial_cash_portfolio = 10000
trade_fee_percent_portfolio = 0.001
max_simultaneous_tickers = 10 # Example: Max 10 tickers at any given time

portfolio_backtest_result = backtest(
    df,
    initial_cash=initial_cash_portfolio,
    trade_fee_percent=trade_fee_percent_portfolio,
    max_open_positions=max_simultaneous_tickers,
    stop_loss_factor=7,
    first_win_factor=3,
    risk=0.05 # Risk parameter still there, but position sizing logic changed
)

# Display the overall portfolio results
print("\n--- Overall Portfolio Backtest Results ---")
for key, value in portfolio_backtest_result.items():
    if key != 'trades': # Don't print the raw trades list here
        print(f"{key}: {value}")

# You can then analyze portfolio_backtest_result['trades'] for individual trade details
combined_trades_df = pd.DataFrame(portfolio_backtest_result['trades'])

# The df_total calculation now applies to the single portfolio backtest result
df_total_portfolio = pd.DataFrame({
    'symbol': ['ALL_SYMBOLS_PORTFOLIO'],
    'tickers_n': [combined_trades_df['symbol'].nunique() if not combined_trades_df.empty else 0],
    'trades_n': [portfolio_backtest_result['total_trades']],
    'win_%': [portfolio_backtest_result['win_rate']],
    'profit_%': [portfolio_backtest_result['total_profit_%']],
    'avg_win': [portfolio_backtest_result['avg_win_profit']],
    'avg_loss': [portfolio_backtest_result['avg_loss_profit']],
    'win_factor': [portfolio_backtest_result['win_factor']],
    'profit': [portfolio_backtest_result['total_profit']],
    'start': [portfolio_backtest_result['starting_cash']],
    'final': [portfolio_backtest_result['final_portfolio_value']]
})


# --- Graphs - Matplotlib tables and graphs ---

# Optional: Round numeric columns for cleaner display
df_summary = df_total_portfolio.round(2)
combined_trades_df.to_html('results_table.html', index=False)
# Create a matplotlib figure and axis
fig, ax = plt.subplots(figsize=(12, len(df_summary) * 0.5 + 1))  # height scales with rows
ax.axis('off')  # turn off axis

# Create the table
table = ax.table(
    cellText=df_summary.values,
    colLabels=df_summary.columns,
    loc='center',
    cellLoc='center',
    colLoc='center'
)

# Optional: Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # scale width, height

# Header row bold
for key, cell in table.get_celld().items():
    row, col = key
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    elif row % 2 == 0:
        cell.set_facecolor('#f1f1f2')
    else:
        cell.set_facecolor('#ffffff')

# Show the table
plt.tight_layout()
plt.show()


# --- Graph for multiple symbols ---

# Group by symbol
# symbols = df['symbol'].unique()
# n = len(symbols)

# # Layout (2 per row is a good balance)
# cols = 2
# rows = (n + cols - 1) // cols

# fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

# for idx, symbol in enumerate(symbols):
#     ax = axes[idx // cols][idx % cols]
#     df_symbol = df[df['symbol'] == symbol]

#     ax.plot(df_symbol['timestamp'], df_symbol['close'], label='Close', color='black')
#     ax.plot(df_symbol['timestamp'], df_symbol['linRegMed'], label='Med', color='red', linestyle='--', linewidth=0.8)
#     ax.plot(df_symbol['timestamp'], df_symbol['linRegHigh'], color='blue', linewidth=1.2)
#     ax.plot(df_symbol['timestamp'], df_symbol['linRegLow'], color='blue', linewidth=1.2)
#     ax.plot(df_symbol['timestamp'], df_symbol['sma10'], color='red', linewidth=1.1)
#     ax.plot(df_symbol['timestamp'], df_symbol['sma20'], color='purple', linewidth=1.1)
#     ax.plot(df_symbol['timestamp'], df_symbol['ema50'], color='green', linewidth=1.1)

#     buy_signals = df_symbol[df_symbol['entryPos'] >= 1]
#     sell_signals = df_symbol[df_symbol['exitPos'] >= 1]

#     ax.scatter(buy_signals['timestamp'], buy_signals['close'], marker='^', color='green', label='Buy', s=50)
#     ax.scatter(sell_signals['timestamp'], sell_signals['close'], marker='v', color='red', label='Sell', s=50)

#     for _, row in buy_signals.iterrows():
#         ax.text(row['timestamp'], row['close'], f"{row['close']:.2f}", color='green', fontsize=6, ha='left', va='bottom')

#     for _, row in sell_signals.iterrows():
#         ax.text(row['timestamp'], row['close'], f"{row['close']:.2f}", color='red', fontsize=6, ha='left', va='top')

#     ax.set_title(f'{symbol}')
#     ax.tick_params(axis='x', rotation=45)
#     ax.grid(True)

# # Hide any unused axes
# for j in range(n, rows * cols):
#     fig.delaxes(axes[j // cols][j % cols])

# # Global legend and labels
# fig.suptitle('Linear Regression Channels with Buy/Sell Signals', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
# plt.show()