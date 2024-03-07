# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:25:51 2023

@author: jrjol
"""

#from polygon_API import get_snp500_companies, list_aggs, get_client
import pandas as pd
from datetime import date
import pypfopt
import numpy as np
import matplotlib.pyplot as plt
#import plotting
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
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Union
import pytz
import requests
from datetime import date
from pypfopt.efficient_frontier import EfficientFrontier


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

from polygon import RESTClient

client = RESTClient(api_key="ocunxnOqC0pnltRqT3VkOiKeCmPE49L7")

def data_extraction(ticker, start_date, end_date):
    
    url = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey=ocunxnOqC0pnltRqT3VkOiKeCmPE49L7".format(
        ticker=ticker, start = start_date, end = end_date)

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        #print(data)
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
    
    results = data["results"]
    df = pd.DataFrame(results)
    df = df[["c"]]
    return df


# How many securities is the client analysing for
n = int(input("Please enter how many securities you are analysing for: "))
tickers = []
for i in range(1,n+1):
    ticker = input("Please enter the ticker for security {}: " .format(i))
    tickers.append(ticker)

end_date = datetime.today().strftime('%Y-%m-%d')
end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

# Calculate one year earlier
start_date_obj = end_date_obj - timedelta(days=365)  # Subtracting 365 days for simplicity

# Convert start_date_obj back to string in 'YYYY-MM-DD' format
start_date = start_date_obj.strftime('%Y-%m-%d')

data = pd.DataFrame()
for ticker in tickers:
    df = data_extraction(ticker, start_date, end_date)
    df.columns = [ticker]
    data = pd.concat([data, df], axis=1)
    
expected_returns = pypfopt.expected_returns.mean_historical_return(data)
cov_matrix = data.cov()

def is_positive_definite(matrix: pd.DataFrame):
      # Convert the pandas DataFrame to a numpy array
      matrix_np = matrix.to_numpy()
      # Check if the matrix is positive definite
      is_pd = np.all(np.linalg.eigvals(matrix_np) > 0)
      return is_pd

z=is_positive_definite(cov_matrix)

ef = EfficientFrontier(
     expected_returns,
     cov_matrix,
     weight_bounds=(0.1,0.25) # We first do this without weight constraints
     )

# Optimize for Max Sharpe ratio
weights = ef.max_sharpe()

companies = list(weights.keys())
values = list(weights.values())

# Creating a bar plot
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.bar(companies, values, color='skyblue')

# Adding labels and title
plt.xlabel('Securities')
plt.ylabel('Percentage of equity invested (%)')
plt.title('Weighting Allocation for Securities')

# Rotating x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Displaying the plot
plt.tight_layout()
plt.show()

values_array = np.array(values)
expected_returns_array = expected_returns.to_numpy()

Proposed_returns = np.dot(values_array, expected_returns_array)
print('Expected returns (per annum) (%): ', round(Proposed_returns*100,1))

percentage_returns = data.pct_change()*100
std_returns = percentage_returns.std()
annualised_volatility = std_returns * np.sqrt(252)
portfolio_std = np.dot(values_array, annualised_volatility)
print('Annualised Std of portfolio: (%)', round(portfolio_std,2))

portfolio_sharpe = (Proposed_returns*100-5.4)/portfolio_std
print('Sharpe ratio: ', round(portfolio_sharpe,2))