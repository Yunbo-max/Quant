# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:59:33 2024

@author: jrjol
"""



"""
Intro

The strategy implemented in the provided code is a quantitative trading strategy designed for the foreign exchange (forex) market. Below is an explanation of the strategy's key components and operations:

Quantitative Approach: The strategy is based on quantitative analysis rather than subjective judgment or fundamental analysis.

Decile Ranking: The strategy ranks currency pairs based on their historical price performance over a specified lookback period (in this case, 12 trading days). It computes compound cumulative returns for each currency pair over the lookback period and assigns decile labels to them.

Long-Short Positioning: The strategy selects a subset of currency pairs for long positions and another subset for short positions based on their decile rankings. It goes long (buy) on the currency pairs that have performed poorly (lowest decile) over the lookback period and goes short (sell) on the currency pairs that have performed well (highest decile).

Equal Weighting: The strategy assumes equal weights for each position. In other words, it allocates the same amount of capital to each selected currency pair.

Trading Mechanism:

The strategy opens positions on the first day/iteration and continues to adjust positions based on changes in decile rankings.
It calculates the profit or loss (PnL) for each trade by comparing the closing prices of the currency pairs at the time of opening and closing the position.
The PnL is calculated based on pip movements, where pip values are determined by the exchange rate and position size.
The strategy keeps track of the account balance over time, reflecting the cumulative PnL from all trades.
Evaluation and Monitoring:

The strategy plots the trades and account balance over time to visualize performance.
It calculates and displays the mean trade size to assess the average profitability of trades.
In summary, this strategy aims to capitalize on mean reversion in currency pairs by taking contrarian positions based on recent price performance. It dynamically adjusts positions based on changes in decile rankings and evaluates performance through PnL and account balance monitoring."""

import yfinance as yf

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import itertools

major_pairs = [
    "EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD"
]

minor_pairs = [
    "EUR/GBP", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD", "EUR/JPY",
    "GBP/JPY", "GBP/AUD", "GBP/CAD", "GBP/NZD", "GBP/CHF",
    "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
    "NZD/JPY", "NZD/CAD", "NZD/CHF",
    "CAD/JPY", "CAD/CHF",
    "CHF/JPY"
]

exotic_pairs = [
    "USD/SGD", "USD/HKD", "USD/TRY", "USD/ZAR", "USD/THB",
    "USD/MXN", "USD/DKK", "USD/NOK", "USD/SEK", "USD/RUB",
    "USD/PLN", "USD/CZK", "EUR/TRY", "EUR/ZAR", "EUR/NOK",
    "EUR/SEK", "EUR/DKK", "EUR/HUF", "EUR/PLN", 
    "GBP/ZAR", "GBP/SGD", "GBP/TRY", 
    "AUD/SGD", "AUD/ZAR",
    "NZD/SGD", 
    "CAD/SGD", 
    "CHF/ZAR", "CHF/SGD"
]
major_pairs = [
    "EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD"
]

# minor pairs - include Korean Won
minor_pairs = [
    "EUR/GBP", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD", "EUR/JPY",
    "GBP/JPY", "GBP/AUD", "GBP/CAD", "GBP/NZD", "GBP/CHF",
    "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
    "NZD/JPY", "NZD/CAD", "NZD/CHF",
    "CAD/JPY", "CAD/CHF",
    "CHF/JPY",
    # 
    "USD/KRW", "EUR/KRW", "GBP/KRW", "AUD/KRW", "CAD/KRW", 
    "CHF/KRW", "NZD/KRW", "JPY/KRW"
]

# exotic currency pairs - exclude Turkish Lira, Russian Ruble, Singapore Dollar
exotic_pairs = [
    "USD/HKD", "USD/ZAR", "USD/THB",
    "USD/MXN", "USD/DKK", "USD/NOK", "USD/SEK", "USD/PLN", "USD/CZK",
    "EUR/ZAR", "EUR/NOK", "EUR/SEK", "EUR/DKK", "EUR/HUF", "EUR/PLN",
    "GBP/ZAR",
    "AUD/ZAR",
    "CHF/ZAR"
]

def format_pairs(pair_list):
    return [f.replace('/', '') + '=X' for f in pair_list]

forex_pairs = format_pairs(major_pairs) + format_pairs(minor_pairs) + format_pairs(exotic_pairs)
# check whether on yfinance
def check_on_yf(pairs):
    
    unavailable_pairs = []
    
    for pair in pairs:
        ticker = yf.Ticker(pair)
        hist = ticker.history(period='5d')
        
        if hist.empty:
            unavailable_pairs.append(pair)
            
    return unavailable_pairs

unav = check_on_yf(forex_pairs)
print(unav)

# fetch 10-year
def fetch_data(pairs, start_date, end_date):
    
    forex_dict = {}

    for pair in pairs:
        ticker = yf.Ticker(pair)
        hist = ticker.history(start=start_date, end=end_date, interval='1d')
        forex_dict[pair] = hist

    return forex_dict


# fetching data from Jan 2014 to Jan 2024

start_date = "2014-01-01"
end_date = "2024-01-24"
forex_data = fetch_data(forex_pairs, start_date, end_date)

closing_prices = pd.DataFrame()

for pair, data in forex_data.items():
    closing_prices[pair] = data['Close']
    
closing_prices.dropna(inplace = True)

forex_df=pd.DataFrame()
forex_df_list = []
for forex_pair, df in forex_data.items():
        # reset index - date index to column
    df_reset = df.reset_index().copy()

    df_selected = df_reset[['Date', 'Close']].copy()
    df_selected.columns = ['dates', 'closing_price']
    df_selected['forex_pair'] = forex_pair

    # append
    forex_df_list.append(df_selected)

# concatenate all df into one
    forex_df = pd.concat(forex_df_list, ignore_index=True)
    forex_df['daily_ret'] = forex_df.groupby('forex_pair')['closing_price'].pct_change()
    forex_df.dropna(inplace = True)
    
# cumulative returns for lookback and holding periods

# 21 trading days ~ one month
LOOKBACK = 21 # cumulative returns for computing and ranking momentum
HOLDING = 21 # cumulative returns for computing returns if the position is held 

#
def compute_cumulative_returns(forex_df, period, lookback = False):
    """
    
    computes compound cumulative returns for a specified t - x to t
    
    """
    # rolling comp cumulative return
    cumulative_returns = forex_df.groupby('forex_pair')['daily_ret'] \
                                 .apply(lambda x: (1 + x).rolling(window=period) \
                                                       .apply(np.prod, raw=True) - 1)

    cumulative_reset = cumulative_returns.reset_index(level=0, drop=True)
    
    if lookback:
        suffex = '_lookback'
    else:
        suffex = '_holding'
        
    cumulative_reset.name = 'cumu_ret' + suffex

    # merge
    forex_cumu = forex_df.merge(cumulative_reset, 
                                left_index=True, right_index=True, how='left')
    
    return forex_cumu


# get cumu for LOOKBACK and HOLDING
forex_cumu = compute_cumulative_returns(forex_df, LOOKBACK, lookback=True)
forex_cumu = compute_cumulative_returns(forex_cumu, HOLDING)

# ranking

GROUPS = 10

def assign_deciles(group, n_groups = 10):
    
    # soft date-group by cumu_ret_lookback
    group = group.sort_values('cumu_ret_lookback', ascending=False)
    
    # assign decile labels
    group['deciles'] = pd.qcut(group['cumu_ret_lookback'], n_groups, labels=False) + 1
    group['deciles'] = 'decile_' + group['deciles'].astype(str)

    return group

forex_rank = forex_cumu.dropna().groupby('dates').apply(assign_deciles, GROUPS).reset_index(drop=True)

# clean index
forex_rank['dates'] = pd.to_datetime(forex_rank['dates'])
forex_rank.set_index('dates', inplace=True)

# HOLDING defined above; resample timeseries to HOLDING time resoluton 
forex_rank_resampled = forex_rank.groupby('forex_pair').resample(f'{HOLDING}D').last().reset_index(level=0, drop=True)

# shift decile assignment, so that decile for t is comptuted at t - HOLDING
forex_rank_resampled['decile_LOOKBACK'] = forex_rank_resampled.groupby('forex_pair')['deciles'].shift(1)

def get_signals(df, n_groups):

    for k in range(1, n_groups + 1):
        df[f'buy signal decile {k}'] = np.where(df['decile_LOOKBACK'] == f'decile_{k}', 1, 0)
        df[f'# positions decile {k}'] = df[df['decile_LOOKBACK'] == f'decile_{k}'].shape[0]

    return df


def get_basket_returns(df, n_groups):

    # assume equal weights
    for k in range(1, n_groups + 1):
        df[f'decile {k} return'] = df[f'buy signal decile {k}'] * df['cumu_ret_holding'] / df[f'# positions decile {k}']
        
    return df

forex_decile_ret = forex_rank_resampled.groupby('dates').apply(get_signals, n_groups = GROUPS)
forex_decile_ret = get_basket_returns(forex_decile_ret, n_groups = GROUPS)

forex_decile_ret.dropna(inplace = True)
forex_decile = forex_decile_ret.droplevel(0)

def aggregate_returns(df, n_groups):

    df_agg = pd.DataFrame()
 
    for k in range(1, GROUPS + 1):
        df_agg[f'decile group {k} return'] = df.groupby(df.index)[f'decile {k} return'].sum()
        
        #
#     df_agg = df_agg.reset_index().drop(columns = ['index'])
    df_agg.index = df.index.unique()
    df_agg.index.name = 'dates'
    df_agg = df_agg.droplevel(0)

    return df_agg

rets = aggregate_returns(forex_decile_ret, n_groups = 10)

cumulative_returns = (1 + rets).cumprod() - 1

plt.figure(figsize=(10, 6))
for column in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)

plt.title("Cumulative Returns of Different Decile Groups Over Time")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

## continuing on trading strat from Subati ############################################

decile_size = 10 # top 10% of performers
lookback=12 # rows lookback = trading days lookback
securities_for_long_short = int(round(len(forex_df_list)*(decile_size/100),0)) # how many pairs does 10% mean?


first_purchase=0 # to start the cross sectional mom, we must open positions on our first day /iteration
PnLs=[] # Each trade PnL through time, eacht trades PnL will be appended to this, you will find we may do 4 or 5 trades on a day.
iteration=[] # to store the dates of each trade
buy_prices = [] # useful for checking that some of the pairs we brought are present in the closing on pair positions
sell_prices = [] # useful for checking with the above 'buy_prices'
init = 0 # account balance starting from 0.

'''
Please note, calculating your ROI for each trade requires you to know the fx rate of your initial investment
to 'go long' on a base curreny. With your inital investment known, then you can match the PnL for a trade, with the
initial investment required to 'go_long' with 1 lot. 1 standard lot converts to 100 000 units

USD/JPY = 100 (here USD is your base currency, and JPY is your quote currency).
When you 'go_long' on an FX pair, you believe the base currency will increase and the quote currency will
decrease in strength. Hence, 'go_long' on USD/JPY=100 means 100 Yen for $1 USD, and as we are long
we would hope USD/JPY>100. If the FX rate increases, then we can get more YEN for $1 USD, 
as the USD has now 'strengthened'.

a single pip is represented by the 4 dp of an FX rate. 1.0001 ---> 1.0005 means a positive 
4 pip movement. (Anything that involves JPY has the 2 dp as the pip number).

When going long or short on a FX pair, you first need to value your pips, secondly, when 
it comes to closing your position, you then calculate the change in pips between 
when your initially went long on the FX pair in comparison to when you are wanting to 
close your position. Remember pip movements refer to the 4 dp of an FX rate movement.

In the code below, I use the variable 'pip_values' to define the pip values when I initially 
went long on the FX pairs, where the FX pairs picked based on which 
base currency (base/quote=rate) has strengthened the most over the last 12 trading days.

As I have 5 long positions opened simulatenously, I trade 1 standard lots for each of them. 
I may close multiple positions simultaneously, I therefore compute the dot product of 
pip values for the pairs I am now going to close my long position of, with the 
pip movement for the corresponding pairs which I now wish to close.


Interestingly, this strategy works by 'going long' of FX pairs which have increased
the most over the past 12 trading days. Interestingly, this seemed to consistently lose 
us money. So, I must inverse my trading stategy by 'going short' instead, this proved to 
give our account balance. In conclusion, any pairs which have performed the best in our basket
over the last 12 days, are overrpiced, and return to their mean (brave conclusion), therefore
we go short. To inverse this stategy (go short on the best performing pairs),
simply add a -ve symbol before the np.dot(.....)
I have already added the -ve symbol in. Take it out if you wish to 'go long'
on the best perfoming pairs. 

Message Joe if anything unclear! More than happy to help!!

Please look through, any issues, please correct and let Joe know.

This code is only a long only FX strat.
'''
Acc_bal_time = []
for i in range(lookback, len(closing_prices)):
    changes_=(closing_prices.iloc[i,:]-closing_prices.iloc[i-lookback,:])/\
        closing_prices.iloc[i-lookback,:] # for each new trading day, the best pairs are recalculated
        
    changes=changes_.sort_values() # descending order rank
    pairs_to_short=list(changes.index[:securities_for_long_short])
    pairs_to_long=list(changes.index[len(forex_df_list)-securities_for_long_short:])
    if first_purchase==0: # open our first 5 trades
        pairs_to_long_copy = pairs_to_long
        first_purchase=1
        long_buy_rates = closing_prices[pairs_to_long].iloc[i,:] #storing the rates we have go long at, so we can calc the pip movement
        pip_values = 0.0001/closing_prices[pairs_to_long].iloc[i,:]*100000 # pip valuation
        
    result = all(value in pairs_to_long for value in pairs_to_long_copy) # checking if the new pairs are same as our current pair open positions
    
    if not result: # new pairs identified
        differences = set(pairs_to_long) ^ set(pairs_to_long_copy) # identifying the different pairs
        differences_list = list(differences) 
        close_long_pairs = set(pairs_to_long_copy) & set(differences_list) # what pairs should be closed check
        close_long_pairs = list(close_long_pairs)
        new_pairs = list(set(pairs_to_long) & set(differences_list)) # what new pairs should be invested in
        if len(close_long_pairs)!=0: 
                
            long_sell_rates = closing_prices[close_long_pairs].iloc[i,:] # rates we sell our positions at
            buy_prices.append(long_buy_rates[close_long_pairs]) 
            sell_prices.append(long_sell_rates)
            # ROI_long = (long_sell_rates-long_buy[close_long_pairs])/\
            #     long_buy[close_long_pairs]
            pip_movement = (long_sell_rates-long_buy_rates[close_long_pairs])*10000 # pip changes of the rate from closing our long position to opening long position
            PnL = -np.dot(pip_values[close_long_pairs], pip_movement) # dot product, for the pip value of each pair, what is the corresponding pip movement
            # pairs_to_long_copy.remove(close_long_pairs)
            PnLs.append(PnL)
            init+=PnL
            Acc_bal_time.append(init)
            
            iteration.append(closing_prices.index[i]) # store dates
            # pairs_to_long_copy.extend(new_pairs)
            
            for N in range(0,len(close_long_pairs)): # removing old pairs, and adding new pairs, whilst valuing our new pair positions
                pip_values = pip_values.drop([close_long_pairs[N]]) # kick out old pair pip values
                pip_values.loc[new_pairs[N]] = 0.0001/\
                    closing_prices[new_pairs[N]].iloc[i]*100000 # add new pair pip values
                long_buy_rates = long_buy_rates.drop([close_long_pairs[N]]) # kick out old pairs
                long_buy_rates[new_pairs[N]] = closing_prices[new_pairs[N]]\
                    .iloc[i] # add in new pairs
        pairs_to_long_copy = pairs_to_long # update our pairs which are now being held, so we can check if they change at the next trading day (row)
   
# # Plotting the equity values over time
plt.plot(iteration, PnLs)
mean_trade_size = sum(PnLs)/len(PnLs)
plt.axhline(y=mean_trade_size, color='red', linestyle='--', label='Mean trade size')

# Adding labels to the axes
plt.xlabel('Date of transaction')
plt.ylabel('Trade PnL')

# Adding a title to the plot
plt.title('Trades vs Time')
plt.legend()
# # Display the plot
plt.show()


plt.plot(iteration, Acc_bal_time)

# Adding labels to the axes
plt.xlabel('Date of transaction')
plt.ylabel('Account balance')

# Adding a title to the plot
plt.title('Account balance vs Time')
plt.legend()
# # Display the plot
plt.show()

    
    
    
    
        

    
    
