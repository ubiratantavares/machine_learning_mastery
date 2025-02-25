import pandas_datareader as pdr
import pandas_datareader.wb

df = pdr.wb.download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
df = df.reset_index()
df = df.filter(["country", "SP.POP.TOTL"])
groups = df.groupby("country")
df = groups.mean()

print(df)
