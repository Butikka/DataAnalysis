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



# Path to files. Up here for convenience.
path_to_files = "C:\\Users\\mustosa\\Documents\\Omat Projektit\\Investing\\data\\hour\\1h\\10b_50b"  # example saving path on device.

# Execute functions that create parameters for the model build and get the DataFrame
df = process_all_functions(path_to_files)

# Get the unique symbols from the DataFrame
df = filter_data_by_symbols(df)

# Calculate entry and exit positions for each symbol
df = entry_exit_for_all_symbols(df)

#Sort the data by date
df = df.sort_values(by=['symbol', 'timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date




# --- Main Backtest function ---
def backtest(df, initial_cash, trade_fee_percent, stop_loss_factor=7, first_win_factor=3, risk=0.05):
    # Initialize symbol-specific variables
    portfolio_cash = initial_cash
    portfolio_stock = 0
    buy_price = 0
    current_symbol = None
    last_trade_date = None

    # Metrics
    total_trades = 0
    total_profit = 0
    total_fees = 0
    win_trades = 0
    max_profit = 0
    max_drawdown = 0
    peak_value = initial_cash
    trades = []
    trade_durations = []
    
    min_cash_to_trade = 1000

    for index, row in df.iterrows():
        close_price = row['close']
        entryPos = row['entryPos']
        exitPos = row['exitPos']
        ticker = row['symbol']
        current_date = row['timestamp']
        volatility = row['volatility']


        # # Calculate first profit level for current row
        # current_take_profit = 1 + (first_win_factor * volatility)



        # Sell condition
        # if portfolio_stock > 0 and (exitPos >= 1 or close_price <= buy_price * current_stop_loss or close_price >= buy_price * current_take_profit):
        if portfolio_stock > 0 and (exitPos >= 1):
            total_value = portfolio_stock * close_price
            shares_sold = portfolio_stock
            fee = total_value * trade_fee_percent
            total_value -= fee
            total_fees += fee
            profit = total_value - (shares_sold * buy_price) - (shares_sold * buy_price * trade_fee_percent)
            portfolio_cash += total_value
            total_profit += profit

            if profit > 0:
                win_trades += 1

            # Update portfolio and metrics after selling
            
            last_trade_date = current_date
            current_symbol = None
            
            trades.append({'date': current_date, 'symbol': ticker, 'type': 'sell', 'price': close_price, 'shares': shares_sold, 'fee': fee, 'profit': profit})
            
            portfolio_stock = 0

            # Update max profit and drawdown calculations
            current_value = portfolio_cash
            if current_value > peak_value:
                peak_value = current_value
            drawdown = (peak_value - current_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
            max_profit = max(max_profit, profit)

        # Buy condition
        if entryPos >= 1 and portfolio_cash >= min_cash_to_trade and portfolio_stock == 0 and \
           (last_trade_date is None or current_date >= last_trade_date) and current_symbol is None:
            position_size = portfolio_cash
            # position_size = (risk * portfolio_cash) / (2 * volatility) # original position sizing
            shares_to_buy = int(position_size / close_price)
            total_cost = shares_to_buy * close_price
            fee = total_cost * trade_fee_percent
            total_cost += fee
            total_fees += fee


            # Update portfolio after buying
            portfolio_stock += shares_to_buy
            portfolio_cash -= total_cost
            buy_price = close_price
            current_symbol = ticker
            last_trade_date = current_date
            stop_loss_price = buy_price * (1 - stop_loss_factor * volatility)
            
            trades.append({'date': current_date, 'symbol': ticker, 'type': 'buy', 'price': close_price, 'shares': shares_to_buy, 'fee': fee, 'stop_loss_price': round(stop_loss_price, 2), 'volatility': volatility})
            total_trades += 1

    # Final calculations for the symbol
    final_portfolio_value = portfolio_cash + (portfolio_stock * df.iloc[-1]['close'])
    total_return = round((final_portfolio_value - initial_cash) / initial_cash * 100, 2)
    win_rate = round(win_trades / total_trades * 100, 2) if total_trades > 0 else 0
    trade_returns = [(t['profit'] / (t['shares'] * buy_price)) * 100 for t in trades if t['type'] == 'sell' and t['shares'] > 0 and buy_price > 0]
    best_trade = round(max(trade_returns, default=0), 2)
    worst_trade = round(min(trade_returns, default=0), 2)
    starting_cash = initial_cash
    bad_perf = win_rate < 50

    return {
        'trades': trades,
        'starting_cash': starting_cash,
        'final_portfolio_value': round(final_portfolio_value, 2),
        'total_profit_%': total_return,
        'total_profit': round(total_profit, 2),
        'total_fees': round(total_fees, 2),
        'total_trades': total_trades,
        'win_rate': win_rate,
        # 'best_trade': best_trade,
        # 'worst_trade': worst_trade,
        'max_profit': round(max_profit, 2),
        'max_drawdown': round(max_drawdown * 100, 2),  # in %
        'bad_perf': bad_perf,
    }


# --- Execute backtesting function ---

# Iterate the backtesting for each symbol in dataframe
# List of symbols to iterate over
symbols = df['symbol'].unique()

# Initialize a list to collect results for each symbol
all_results = []
all_trades = []

for symbol in symbols:
    # Filter the DataFrame to only include data for the current symbol
    df_symbol = df[df['symbol'] == symbol]
    
    # Run backtest for the current symbol
    result = backtest(df_symbol, initial_cash=10000, trade_fee_percent=0.001)
    
    # Store the result in the list
    all_results.append({
        'symbol': symbol,
        'start': result['starting_cash'],
        'end': result['final_portfolio_value'],
        'profit_%': result['total_profit_%'],
        'profit_eur': result['total_profit'],
        'trades_n': result['total_trades'],
        'fees': result['total_fees'],
        'win_%': result['win_rate'],
        # 'best': result['best_trade'],
        # 'worst': result['worst_trade'],
        # 'avg_trade_duration': result['avg_trade_duration'],
        'max_eur': result['max_profit'],
        'drawdown_%': result['max_drawdown'],
        'bad_perf': result['bad_perf'],
    })
    
    # Store trade details separately for detailed analysis
    trades_df = pd.DataFrame(result['trades'])
    trades_df['symbol'] = symbol  # Add symbol to each trade for identification
    all_trades.append(trades_df)

# # Testing dataframe
# df = pd.concat(all_trades, ignore_index=True)
# df = df[df['type'] == 'sell']
# # print(df.columns.tolist())
# print(df.tail(10))

# Combine all trades data into one DataFrame
combined_trades_df = pd.concat(all_trades, ignore_index=True)

# Convert results to a DataFrame for easier analysis
combined_results_df = pd.DataFrame(all_results)

df_total = pd.DataFrame({
    'symbol': ['ALL_SYMBOLS'],
    'tickers_n': [combined_results_df['symbol'].count()],
    'trades_n': [combined_results_df['trades_n'].sum()],
    'win_%': [((combined_trades_df['profit'] > 0).sum() / combined_trades_df['profit'].count()) * 100 if combined_trades_df['profit'].count() > 0 else 0],
    'avg_profit_%': [combined_results_df['profit_%'].mean()],
    'avg_win': [combined_trades_df.loc[combined_trades_df['profit'] > 0, 'profit'].mean()],
    'avg_loss': [combined_trades_df.loc[combined_trades_df['profit'] < 0, 'profit'].mean()],
    'win_factor': [combined_trades_df.loc[combined_trades_df['profit'] > 0, 'profit'].mean() / abs(combined_trades_df.loc[combined_trades_df['profit'] < 0, 'profit'].mean()) if combined_trades_df.loc[combined_trades_df['profit'] < 0, 'profit'].count() > 0 else 0],
    'profit': [combined_results_df['profit_eur'].sum()],
    'start': [combined_results_df['start'].sum()],
    'end': [combined_results_df['end'].sum()]
})



# --- Graphs - Matplotlib Tables and graphs ---

# Optional: Round numeric columns for cleaner display
df_summary = df_total.round(2)
combined_results_df.to_html('results_table.html', index=False)
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


# --- Graph for one symbol ---

# Plotting
# Plot the base price chart
# plt.plot(df['timestamp'], df['close'], label='Close', color='black')
# plt.plot(df['timestamp'], df['linRegMed'], label='Med', color='grey', linestyle='--', linewidth=0.8)
# plt.plot(df['timestamp'], df['linRegHigh'], color='blue', linewidth=1.2)
# plt.plot(df['timestamp'], df['linRegLow'], color='blue', linewidth=1.2)
# plt.plot(df['timestamp'], df['sma10'], color='red', linewidth=1.1)
# plt.plot(df['timestamp'], df['sma20'], color='purple', linewidth=1.1)

# buy_signals = df[df['entryPos'] >= 1]
# sell_signals = df[df['exitPos'] >= 1]

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
#     # ax.plot(df_symbol['timestamp'], df_symbol['sma10'], color='red', linewidth=1.1)
#     # ax.plot(df_symbol['timestamp'], df_symbol['sma20'], color='purple', linewidth=1.1)

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