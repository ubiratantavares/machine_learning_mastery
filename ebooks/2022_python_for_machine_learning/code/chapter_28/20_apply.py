import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    minyear = sum_series[year_indices].astype(float).idxmin()
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
print(df_years)
