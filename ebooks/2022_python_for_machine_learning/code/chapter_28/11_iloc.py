import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_r5 = df.iloc[5:11]
df_c1_r5 = df.iloc[5:11, 1:7]
