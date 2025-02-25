import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
print(df_sum)
