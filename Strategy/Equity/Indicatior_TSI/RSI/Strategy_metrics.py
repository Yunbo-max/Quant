# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-16 13:46:11
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-02 10:38:59


import numpy as np

class StrategyStatistics:
    def __init__(self, trades, wins, losses, profits, losses_amount, win_amounts, fees):
        self.trades = trades
        self.wins = wins
        self.losses = losses
        self.profits = profits
        self.losses_amount = losses_amount
        self.win_amounts = win_amounts
        self.fees = fees

    def total_trades(self):
        return self.trades

    def average_win(self):
        if self.wins == 0:
            return 0
        return sum(self.win_amounts) / self.wins

    def average_loss(self):
        if self.losses == 0:
            return 0
        return sum(self.losses_amount) / self.losses

    def compounding_annual_return(self, initial_capital, final_capital, years):
        return (final_capital / initial_capital) ** (1 / years) - 1

    def drawdown(equity_curve):
        highest_equity = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - highest_equity) / highest_equity
        max_drawdown = np.max(drawdown)
        return max_drawdown

        
        return max_drawdown

    def expectancy(self):
        if self.trades == 0:
            return 0
        return (sum(self.profits) - sum(self.losses_amount)) / self.trades

    def net_profit(self):
        return sum(self.profits) - sum(self.fees)

    def sharpe_ratio(self, returns, risk_free_rate):
        annual_return = np.mean(returns) * 252
        annual_std = np.std(returns) * np.sqrt(252)
        if annual_std == 0:
            return 0
        return (annual_return - risk_free_rate) / annual_std

    def sortino_ratio(self, returns, risk_free_rate):
        downside_returns = returns[returns < 0]
        annual_return = np.mean(returns) * 252
        downside_std = np.std(downside_returns) * np.sqrt(252)
        if downside_std == 0:
            return 0
        return (annual_return - risk_free_rate) / downside_std

    def probabilistic_sharpe_ratio(self, returns, risk_free_rate):
        downside_returns = returns[returns < risk_free_rate]
        p_mean = np.mean(returns - risk_free_rate)
        p_std = np.std(downside_returns)
        if p_std == 0:
            return 0
        return p_mean / p_std

    def loss_rate(self):
        if self.trades == 0:
            return 0
        return self.losses / self.trades

    def win_rate(self):
        if self.trades == 0:
            return 0
        return self.wins / self.trades

    def profit_loss_ratio(self):
        if sum(self.losses_amount) == 0:
            return 0
        return sum(self.win_amounts) / sum(self.losses_amount)

    def alpha(self, returns, benchmark_returns, risk_free_rate):
        beta = self.beta(returns, benchmark_returns)
        annual_return = np.mean(returns) * 252
        benchmark_annual_return = np.mean(benchmark_returns) * 252
        return annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))

    def beta(self, returns, benchmark_returns):
        cov_matrix = np.cov(returns, benchmark_returns)
        return cov_matrix[0][1] / cov_matrix[1][1]

    def annual_standard_deviation(self, returns):
        return np.std(returns) * np.sqrt(252)

    def annual_variance(self, returns):
        return self.annual_standard_deviation(returns) ** 2

    def information_ratio(self, returns, benchmark_returns):
        active_return = returns - benchmark_returns
        tracking_error = np.std(active_return)
        if tracking_error == 0:
            return 0
        return np.mean(active_return) / tracking_error

    def tracking_error(self, returns, benchmark_returns):
        active_return = returns - benchmark_returns
        return np.std(active_return)

    def treynor_ratio(self, returns, benchmark_returns, risk_free_rate):
        beta = self.beta(returns, benchmark_returns)
        return (np.mean(returns) - risk_free_rate) / beta

    def total_fees(self):
        return sum(self.fees)

    def estimated_strategy_capacity(self, trading_capacity, trading_frequency):
        return trading_capacity / trading_frequency

    def lowest_capacity_asset(self, assets_capacities):
        return min(assets_capacities)

    def portfolio_turnover(self, trades, total_assets):
        return trades / total_assets

