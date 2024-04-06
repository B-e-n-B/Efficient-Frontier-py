import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

#///////////////////////////THIS IS WHERE CHANGES CAN BE EASILY MADE////////////////////////////////////////////////////////

# These are portfolio weights 
# Index 0 = Other Symbol - Index 1 = Equities Symbol - Index 2 = Fixed Income Symbol 
market_weights = [0.1, 0.6, 0.3]  #Portfolio 1
market_weights_1 = [0.1, 0.54, 0.36]  #Portfolio 2
market_weights_2 = [0.1, 0.5, 0.4]  #Portfolio 3


#These are xytext positions
#Change them to change where text (describing portfolio) is relative to respective portfolio displayed on chart

x1 = -10  #Portfolio 1
y1 = 25   

x2 = 15  #Portfolio 2
y2 = 95  

x3 = -40  #Portfolio 3
y3= 55  


#This Changes Color of marker
#Can change from rgb value to name of a color Ex: (0.1, 0.6, 0.3) -> 'green'
#You might notice that I made the RGB values the same as portfolio weights just for gits and shiggles

color1 = (0.1, 0.6, 0.3)  #Portfolio 1
color2 = (0.1, 0.6, 0.3)  #Portfolio 2
color3 =(0.1, 0.5, 0.4)  #Portfolio 3

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




ticker1 = input("Enter Equities Symbol: ")
ticker2 = input("Enter Fixed Income Symbol: ")
ticker3 = input("enter other symbol (etf, stock, etc): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")
Itterations = int(input("How many Itterations (how many portfolios to simulate)? "))


# YES, I was too lazy to change varriable names ¯\_(ツ)_/¯

# Bring in data
obsidiandataf = yf.download(ticker2, start=start_date, end=end_date)
spydataf = yf.download(ticker1, start=start_date, end=end_date)
aggdataf = yf.download(ticker3, start=start_date, end=end_date)


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

if markeroptimized_weights[0] < .1:
    markeroptimized_weights[0] = 0
if markeroptimized_weights[1] < .1:
    markeroptimized_weights[1] = 0
if markeroptimized_weights[2] < .1:
    markeroptimized_weights[2] = 0

# Calculate optimized portfolio performance
markeroptimized_return, markeroptimized_std_dev = markercalculate_portfolio_performance(
    markeroptimized_weights, markerreturns, markercov_matrix)

# Generate random portfolios
markernum_portfolios = Itterations
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
plt.scatter(market_std_dev, market_return, marker='D', color=color1, s=100, label='Portfolio 1')
plt.annotate(f'{market_weights[1]}% {ticker1}, {market_weights[2]}% {ticker2}, \n{market_weights[0]}% {ticker3}', (market_std_dev, market_return), textcoords="offset points", xytext=(x1,y1), ha='center', 
arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.2', color='grey'))

plt.scatter(market_std_dev_1, market_return_1, marker='D', color=color2, s=100, label='Portfolio 2')
plt.annotate(f'54% {ticker1}, 36% {ticker2}, \n10% {ticker3}', (market_std_dev_1, market_return_1), textcoords="offset points", xytext=(x2,y2), ha='center', 
arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.2', color='grey'))

plt.scatter(market_std_dev_2, market_return_2, marker='D', color=color3, s=100, label='Portfolio 3')
plt.annotate(f'50% {ticker1}, 40% {ticker2}, \n10% {ticker3}', (market_std_dev_2, market_return_2), textcoords="offset points", xytext=(x3,y3), ha='center', 
arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.2', color='grey'))
plt.xlabel('Volatility (Risk)')
plt.ylabel('Annual Return')
plt.legend()
plt.show()