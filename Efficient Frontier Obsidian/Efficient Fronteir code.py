import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import os


# Get the current working directory
current_directory = os.getcwd()

# Construct the file path relative to the current directory
obs = os.path.join(current_directory, 'obsidiandata.csv')
spy = os.path.join(current_directory, 'spydata.csv')
agg = os.path.join(current_directory, 'aggdata.csv')

# Bring in data
obsidiandataf = pd.read_csv(obs)
spydataf = pd.read_csv(spy)
aggdataf = pd.read_csv(agg)

# Define the 'close' rows of each dataframe
obsidiandataf_prices = obsidiandataf['Close']
spydataf_prices = spydataf['Close']
aggdataf_prices = aggdataf['Close']

markerdata = pd.concat([obsidiandataf_prices, spydataf_prices, aggdataf_prices], axis=1)
markerdata.columns = ['Obsidian Performance', 'SPY', 'AGG']

# Calculate returns and covariance matrix
markerreturns = markerdata.pct_change().dropna()
markercov_matrix = markerreturns.cov()

# Function to calculate portfolio markerreturns and volatility
def markercalculate_portfolio_performance(markerweights, markerreturns, markercov_matrix):
    markerweights = np.array(markerweights)  # Convert markerweights to a NumPy array
    markerportfolio_return = np.sum(markerreturns.mean() * markerweights) * 252
    markerportfolio_std_dev = np.sqrt(np.dot(markerweights.T, np.dot(markercov_matrix, markerweights))) * np.sqrt(252)
    return markerportfolio_return, markerportfolio_std_dev

# Function to minimize the negative portfolio Sharpe ratio
def markerminimize_negative_sharpe_ratio(markerweights, markerreturns, markercov_matrix, risk_free_rate):
    markerportfolio_return, markerportfolio_std_dev = markercalculate_portfolio_performance(markerweights, markerreturns, markercov_matrix)
    markersharpe_ratio = (markerportfolio_return - risk_free_rate) / markerportfolio_std_dev
    return -markersharpe_ratio

# Set the risk-free rate
risk_free_rate = 0.02

# Define optimization constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Define optimization bounds
bounds = tuple((0, 1) for _ in range(len(markerdata.columns)))

# Initial guess for markerweights
markerinitial_weights = [1/len(markerdata.columns)] * len(markerdata.columns)  # Equal allocation

# Optimize portfolio markerweights for maximum Sharpe ratio
markerresult = minimize(markerminimize_negative_sharpe_ratio, markerinitial_weights,
                  args=(markerreturns, markercov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Get optimized markerweights
markeroptimized_weights = markerresult.x

# Calculate optimized portfolio performance
markeroptimized_return, markeroptimized_std_dev = markercalculate_portfolio_performance(
    markeroptimized_weights, markerreturns, markercov_matrix)

# Generate random portfolios
markernum_portfolios = 200
markerall_weights = np.zeros((markernum_portfolios, len(markerdata.columns)))
markerret_arr = np.zeros(markernum_portfolios)
markervol_arr = np.zeros(markernum_portfolios)

for i in range(markernum_portfolios):
    markerweights = np.random.random(len(markerdata.columns))
    markerweights /= np.sum(markerweights)
    markerall_weights[i, :] = markerweights
    markerret_arr[i], markervol_arr[i] = markercalculate_portfolio_performance(markerweights, markerreturns, markercov_matrix)

# Plot efficient frontier
plt.figure(figsize=(12, 8))
plt.plot(markervol_arr, markerret_arr, 'o', markersize=5, label='Portfolios')

# Calculate performance of market portfolios
market_weights = [0.1, 0.6, 0.3]
market_weights_1 = [0.1, 0.54, 0.36]
market_weights_2 = [0.1, 0.5, 0.4]

market_return, market_std_dev = markercalculate_portfolio_performance(market_weights, markerreturns, markercov_matrix)
market_return_1, market_std_dev_1 = markercalculate_portfolio_performance(market_weights_1, markerreturns, markercov_matrix)
market_return_2, market_std_dev_2 = markercalculate_portfolio_performance(market_weights_2, markerreturns, markercov_matrix)


# Plot markers and add annotations to each marker
plt.scatter(markeroptimized_std_dev, markeroptimized_return, marker='D', color='Orange', s=100, label='Optimized Portfolio')
plt.annotate('60% Stocks, 30% Bonds, \n10% Obsidian', (markeroptimized_std_dev, markeroptimized_return), textcoords="offset points", xytext=(-10,25), ha='center', 
arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.2', color='grey'))

plt.scatter(market_std_dev_1, market_return_1, marker='D', color=(0.1, 0.6, 0.3), s=100, label='Common Stock/Bond blend + Obsidian')
plt.annotate('54% Stocks, 36% Bonds, \n10% Obsidian', (market_std_dev_1, market_return_1), textcoords="offset points", xytext=(15,95), ha='center', 
arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.2', color='grey'))

plt.scatter(market_std_dev_2, market_return_2, marker='D', color=(0.1, 0.5, 0.4), s=100, label='Common Bond/Stock Blend + Obsdian')
plt.annotate('50% Stocks, 40% Bonds, \n10% Obsidian', (market_std_dev_2, market_return_2), textcoords="offset points", xytext=(-40,55), ha='center', 
arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.2', color='grey'))
plt.xlabel('Annual Return')
plt.ylabel('Volatility (Risk)')
plt.legend()
plt.show()