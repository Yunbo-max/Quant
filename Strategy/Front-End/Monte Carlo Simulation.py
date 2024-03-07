# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-03-07 09:57:46
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-07 10:10:33


"""This Python script appears to implement a Monte Carlo simulation for evaluating the performance of a stock portfolio based on Mean-Variance Portfolio Theory (MPT). Here's a breakdown of the key components and operations:

Data Retrieval: Historical price data for selected stocks (in this case, AAPL, MSFT, and AMZN) is obtained using the Yahoo Finance API (yfinance library). The data is retrieved for a specified time period.

Portfolio Setup:

The script sets up the portfolio with selected stocks (portfolio variable).
Initial weights for each stock are randomly assigned (weights variable) and normalized to ensure they sum up to 1.
Mean and Covariance Calculation: The mean returns and covariance matrix of the selected stocks are calculated using historical price data. These are essential inputs for MPT.

Monte Carlo Simulation:

The script conducts a Monte Carlo simulation to forecast the performance of the portfolio over a specified time horizon (T). The simulation is repeated mc_sims times.
For each simulation:
Random normal variables (Z) are generated to simulate daily returns for each stock. These random variables are independent.
A Cholesky decomposition of the covariance matrix is performed to generate correlated daily returns for the portfolio.
The portfolio value is calculated based on the cumulative product of the weighted sum of daily returns.
The resulting portfolio values for each simulation are stored in portfolio_sims.
Visualization: The script plots the portfolio values over the simulation period (Days) to visualize the potential portfolio performance under different scenarios.

Risk Metrics (commented out):

The script includes functions for calculating Value at Risk (VaR) and Conditional Value at Risk (CVaR) for the simulated portfolio returns. However, these calculations are currently commented out.
The overall goal of this script is to assess the potential future performance of a stock portfolio using a Monte Carlo simulation approach, taking into account historical returns and their covariance structure. The visualization provides insights into the range of possible outcomes for the portfolio over the specified time horizon.
"""




import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import yfinance as yf
import matplotlib.pyplot as plt

# from pandas_datareader import data as pdr

# import data
def get_data(stocks, start, end):
    df = yf.download(stocks, start, end)   
    df = df.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1) # Only extract data for CLOSE
    returns = df.pct_change() # Get return by taking difference in price for every time stamp
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Set up portfolio
portfolio = ['AAPL', 'MSFT', 'AMZN']

# Set up weights according to MPT (randomised now)
weights = np.random.random(len(portfolio))
weights /= np.sum(weights)

# Setting start date as 300 days before today
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=300)

meanReturns, covMatrix = get_data(portfolio, startDate, endDate)



# Monte Carlo Method
mc_sims = 400 # number of simulations
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
    L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z) #Correlated daily returns for individual stocks
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

# def mcVaR(returns, alpha=5):
#     """ Input: pandas series of returns
#         Output: percentile on return distribution to a given confidence level alpha
#     """
#     if isinstance(returns, pd.Series):
#         return np.percentile(returns, alpha)
#     else:
#         raise TypeError("Expected a pandas data series.")

# def mcCVaR(returns, alpha=5):
#     """ Input: pandas series of returns
#         Output: CVaR or Expected Shortfall to a given confidence level alpha
#     """
#     if isinstance(returns, pd.Series):
#         belowVaR = returns <= mcVaR(returns, alpha=alpha)
#         return returns[belowVaR].mean()
#     else:
#         raise TypeError("Expected a pandas data series.")


# portResults = pd.Series(portfolio_sims[-1,:])

# VaR = initialPortfolio - mcVaR(portResults, alpha=5)
# CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

# print('VaR_5 ${}'.format(round(VaR,2)))
# print('CVaR_5 ${}'.format(round(CVaR,2)))
