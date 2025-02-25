import pandas_datareader as pdr

# Reading Apple shares from Yahoo Finance server
shares_df = pdr.DataReader('AAPL', 'yahoo', start='2021-01-01', end='2021-12-31')
# Look at the data read
print(shares_df)
