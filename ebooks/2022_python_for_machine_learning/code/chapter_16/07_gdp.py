from pandas_datareader.wb import WorldBankReader

gdp = WorldBankReader("NY.GDP.MKTP.CN", "SE", start=1960, end=2020).read()
