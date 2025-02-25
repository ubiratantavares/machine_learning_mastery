import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
df_pivot = df_all_co.pivot_table(index="State", columns="Tier 1 Description", values="emissions21")
df_melt = df_pivot.melt(value_name="emissions 2021", var_name="Tier 1 Description", ignore_index=False)
print(df_melt)
