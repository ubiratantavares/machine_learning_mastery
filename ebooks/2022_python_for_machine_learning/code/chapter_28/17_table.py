import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
print(df_2021)
