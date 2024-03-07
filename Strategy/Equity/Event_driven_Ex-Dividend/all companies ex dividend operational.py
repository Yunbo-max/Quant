# -*- coding: utf-8 -*-

"""
Created on Sun Nov  5 00:00:37 2023

@author: jrjol

This code extracts all the current SPY equities. It then loops through them and extracts their exdividend dates. 
Then for each company and each of their corresponding ex dividend dates, it attempts to monitor the price reaction 
around the exdividend date. for i in tickerzz loop is primarily where you can edit. When I try to calculate roi, I 
refer to 'data'. 'data' is a dataframe of all the stock prices within a specified range, change it, try new ideas.
Remember you can also change what data you want, minutes, seconds, etc, but you will then have to change the buy and 
sell locations again. In 'data' column 3 refers to the close, which is when I am closing my long position. '0' refers to
the open.
look at the following variables: 'success_rate' will tell you how many positive ROI trades we made vs negatives as a ratio.
'average' simply tells you the average roi per trade. This code does not include trading fees or spread costs!!
Try running this code for a different open and closing times, let me know if you see average>0.002 or average<-0.002
"""


'''
Introduction

Identifying Ex-Dividend Dates: The strategy involves identifying stocks that are about to go ex-dividend. On the ex-dividend date, the price of the stock typically drops by an amount roughly equivalent to the dividend to be paid out.

Buying Before Ex-Dividend Date: Traders using this strategy often buy the stock a few days before the ex-dividend date. They aim to capture the dividend payment while hoping for a potential price increase leading up to the ex-dividend date.

Selling After Ex-Dividend Date: After the ex-dividend date, the stock price may recover, or traders may sell the stock to capture the dividend and any potential price increase. However, some traders may hold the stock for longer periods if they anticipate further price appreciation.

Calculating ROI: The code calculates the Return on Investment (ROI) for each trade based on the difference in stock prices before and after the ex-dividend date. Positive ROI indicates a profitable trade, while negative ROI indicates a loss.

Success Rate and Average ROI: The success rate and average ROI are calculated to evaluate the effectiveness of the strategy. A high success rate and positive average ROI would indicate that the strategy is profitable over the specified time period.

Overall, this strategy aims to capitalize on the temporary price drop that often occurs after a stock goes ex-dividend, while also considering potential price movements leading up to and following the ex-dividend date. It's important to note that like any trading strategy, there are risks involved, and past performance may not necessarily predict future results.
'''






from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import pytz
from polygon import RESTClient
from polygon.rest.models import (
    Agg,
    DailyOpenCloseAgg,
    GroupedDailyAgg,
    PreviousCloseAgg,
    Sort,
)
from polygon.rest.models.request import RequestOptionBuilder
from urllib3 import HTTPResponse

import yfinance as yf
import pandas as pd

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def get_client():
    return RESTClient("VoUAUxVML9mwSpopBIK7vPjIF_Y25Yk5")


# TODO: Account for market holidays and half-days
# TODO: Determine timezone from stock/equity ticker
def within_trading_hours(
    timestamp: Union[str, int, datetime, date], timezone: str
) -> bool:
    """
    Determine whether a given instant is within trading hours for a particular exchange.

    :param timestamp: The timestamp (in milliseconds) to be checked.
    :param timezone: The timezone to check trading hours in, as a IANA Time Zone Database Name.
    :return: True if the timestamp is within trading hours, False otherwise.
    """
    dt = datetime.fromtimestamp(timestamp / 1000, tz=pytz.timezone(timezone))
    if dt.weekday() < 5:
        opening_time = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        closing_time = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        return opening_time <= dt <= closing_time
    else:
        return False


# TODO: Determine timezone from stock/equity ticker
def parse_timestamp(timestamp: Union[str, int, datetime, date], timezone: str) -> str:
    """
    Parse a timestamp to a readable format for easy comparisons.

    :param timestamp: The timestamp (in milliseconds) to be formatted.
    :param timezone: The timezone to check trading hours in, as a IANA Time Zone Database Name.
    :return: Date-time representation in the following format: %Y-%m-%d %H:%M:%S %Z.
    """
    dt = datetime.fromtimestamp(timestamp / 1000, tz=pytz.timezone(timezone))
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def agg_to_dict(agg: Agg) -> dict[str, Union[None, float, int, bool]]:
    """
    Convert an aggregate object to a dictionary, and add any extra data that may be useful.

    :param agg: The aggregate object to parse.
    :return: Dictionary with keys corresponding to attributes of the object.
    """
    return {
        "open": agg.open,
        "high": agg.high,
        "low": agg.low,
        "close": agg.close,
        "volume": agg.volume,
        "vwap": agg.vwap,
        "timestamp": agg.timestamp,
        "datetime": parse_timestamp(agg.timestamp, "America/New_York"),
        "transactions": agg.transactions,
        "otc": agg.otc,
    }


def list_aggs(
    client: RESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    # "from" is a keyword in python https://www.w3schools.com/python/python_ref_keywords.asp
    from_: Union[str, int, datetime, date],
    to: Union[str, int, datetime, date],
    include_extended_hours: bool = False,
    adjusted: Optional[bool] = None,
    sort: Optional[Union[str, Sort]] = None,
    limit: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    raw: bool = False,
    options: Optional[RequestOptionBuilder] = None,
) -> pd.DataFrame:
    """
    List aggregate bars for a ticker over a given date range in custom time window sizes.

    :param client: The RESTClient object to perform the request with.
    :param ticker: The ticker symbol.
    :param multiplier: The size of the timespan multiplier.
    :param timespan: The size of the time window.
    :param from_: The start of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param to: The end of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param include_extended_hours: True if pre-market and after-hours trading data are to be included, False otherwise.
    :param adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    :param sort: Sort the results by timestamp. asc will return results in ascending order (oldest at the top), desc will return results in descending order (newest at the top).The end of the aggregate time window.
    :param limit: Limits the number of base aggregates queried to create the aggregate results. Max 50000 and Default 5000. Read more about how limit is used to calculate aggregate results in Polygon's on Aggregate Data API Improvements.
    :param params: Any additional query params.
    :param raw: Return raw object instead of results object.
    :return: Pandas DataFrame representation of aggregate objects.
    """
    aggs = []
    for a in client.list_aggs(
        ticker,
        multiplier,
        timespan,
        from_,
        to,
        adjusted=adjusted,
        sort=sort,
        limit=limit,
        params=params,
        raw=raw,
        options=options,
    ):
        if include_extended_hours or within_trading_hours(
            a.timestamp, "America/New_York"
        ):
            aggs.append(agg_to_dict(a))
    return pd.DataFrame(aggs)



url = 'https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', {'id': 'constituents'})
tickerzz = []
for row in table.find_all('tr')[1:]:
    columns = row.find_all('td')
    ticker = columns[0].text.strip()
    tickerzz.append(ticker)

start = '2020-09-5'
end = '2023-10-29'
dictionary={}
data = pd.DataFrame()
for i in tickerzz:
    dividends = yf.Ticker(i).dividends

    if isinstance(dividends, pd.DataFrame) or isinstance(dividends, pd.Series):
        series = dividends.loc[start:end]
        series = series.rename(i)
        dictionary[i]=series
        data = pd.concat([data, series], axis=1)

result_dict = {}

for company_ticker, column in data.items():
    company_data = {}
    for index, row in column.items():
        if not pd.isna(row):
            
            company_data[index] = row
    company_series = pd.Series(company_data, name=company_ticker)
    result_dict[company_ticker] = company_series

All_ROIs=[]
positive=0
negative=0

#tickerzz = [ticker for ticker in tickerzz if ticker not in ['BRK.B', 'BF.B']]
ticker_series = pd.Series(tickerzz)
ticker_series = ticker_series[~ticker_series.isin(['BRK.B', 'BF.B'])]
tickerzz = ticker_series.tolist()
for i in tickerzz:
    company_exdividend_dates = result_dict[i]
    Numbers_of_dates_to_check = len(company_exdividend_dates)

    for ex_date in range(Numbers_of_dates_to_check):  # Iterate within the bounds!!
        date_index = company_exdividend_dates.index[ex_date]
        new_date = date_index + timedelta(days=5)
        new_date = new_date.strftime("%Y-%m-%d")
        date_index = date_index.strftime("%Y-%m-%d")
        stock_data = list_aggs(
            get_client(), i, 1, "minute", str(date_index), str(new_date), limit=50000
        )
        if not stock_data.empty:
            roi = (stock_data.iloc[131, 3] - stock_data.iloc[124, 0]) / stock_data.iloc[124, 0]
            All_ROIs.append(roi)
            if roi > 0:
                positive += 1
            else:
                negative += 1
all_trades=positive+negative
success_rate=positive/all_trades
average=sum(All_ROIs)/all_trades