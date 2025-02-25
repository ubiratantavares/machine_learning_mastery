import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
print(df_joined)
