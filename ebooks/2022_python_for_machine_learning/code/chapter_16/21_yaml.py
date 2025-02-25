import warnings
warnings.simplefilter("ignore")

from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
import pandas as pd
import yaml

# Load config from YAML file
with open("config.yaml", "r") as fp:
    args = yaml.safe_load(fp)

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
