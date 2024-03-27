# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-03-23 11:32:08
# @Last Modified by:   Yunbo
# @Lastimport numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Given data
expected_returns = np.array([0.10, 0.15])  # Expected returns for Stock A and Stock B
covariance_matrix = np.array([[0.0144, 0.0144], [0.0144, 0.04]])  # Covariance matrix
risk_free_rate = 0.03  # Risk-free rate

# Simulate different portfolio weights
num_portfolios = 100
portfolio_weights = np.linspace(0, 1, num_portfolios)
portfolio_returns = []
portfolio_volatilities = []

# Calculate portfolio returns and volatilities
for w_A in portfolio_weights:
    w_B = 1 - w_A
    portfolio_return = w_A * expected_returns[0] + w_B * expected_returns[1]
    portfolio_returns.append(portfolio_return)
    portfolio_volatility = np.sqrt((w_A ** 2) * covariance_matrix[0, 0] +
                                   (w_B ** 2) * covariance_matrix[1, 1] +
                                   2 * w_A * w_B * covariance_matrix[0, 1])
    portfolio_volatilities.append(portfolio_volatility)

# Calculate Sharpe ratios
sharpe_ratios = (np.array(portfolio_returns) - risk_free_rate) / np.array(portfolio_volatilities)

# Find the portfolio with the maximum Sharpe ratio
max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe_ratio = sharpe_ratios[max_sharpe_idx]
optimal_portfolio_return = portfolio_returns[max_sharpe_idx]
optimal_portfolio_volatility = portfolio_volatilities[max_sharpe_idx]
optimal_portfolio_weight_A = portfolio_weights[max_sharpe_idx]
optimal_portfolio_weight_B = 1 - optimal_portfolio_weight_A

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratios, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_portfolio_volatility, optimal_portfolio_return, marker='*', color='red', s=100, label='Optimal Portfolio')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)
plt.show()

# Print the optimal portfolio weights and Sharpe ratio
print(f"Optimal Portfolio Weights - Stock A: {optimal_portfolio_weight_A:.2f}, Stock B: {optimal_portfolio_weight_B:.2f}")
print(f"Optimal Portfolio Expected Return: {optimal_portfolio_return:.2f}")
print(f"Optimal Portfolio Volatility: {optimal_portfolio_volatility:.2f}")
print(f"Optimal Portfolio Sharpe Ratio: {max_sharpe_ratio:.2f}")
