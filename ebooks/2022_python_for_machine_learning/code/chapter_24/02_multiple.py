import pandas_datareader as pdr

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo',
                                    start='2021-01-01', end='2021-12-31')
print(shares_multiple_df.head())
