import yfinance as yf
import pandas as pd
import numpy as np

ticker = "AAPL"
start = "2000-02-18"
end = "2023-02-18"
df = yf.download(ticker, start, end, interval="1d")

print(f"The stock is {ticker}")
df
