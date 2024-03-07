# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-03-07 15:21:35
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-07 16:04:53
import datetime
import pandas as pd
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover
from Polygon.data import *
import pandas_ta as ta

SPY = list_aggs(get_client(), "SPY", 1, "day", "2000-01-01", "2018-12-31", limit=50000)

SPY.columns = SPY.columns.str.capitalize()


class RsiOscillator(Strategy):

    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    # Do as much initial computation as possible
    def init(self):
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), self.rsi_window)

    # Step through bars one by one
    # Note that multiple buys are a thing here
    def next(self):
        if crossover(self.rsi, self.upper_bound):
            self.position.close()
        elif crossover(self.lower_bound, self.rsi):
            self.buy()


bt = Backtest(SPY, RsiOscillator, cash=10_000, commission=.002)
stats = bt.run()
bt.plot()
