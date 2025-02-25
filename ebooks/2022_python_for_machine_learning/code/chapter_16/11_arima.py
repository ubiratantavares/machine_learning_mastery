from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
import pandas as pd

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--country", default="SE", help="Two-letter country code")
parser.add_argument("-l", "--length", default=40, type=int, help="Length of time series to fit the ARIMA model")
parser.add_argument("-s", "--start", default=0, type=int, help="Starting offset to fit the ARIMA model")
args = vars(parser.parse_args())

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df)
