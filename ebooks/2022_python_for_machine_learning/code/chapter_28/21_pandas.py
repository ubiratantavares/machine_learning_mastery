import pandas as pd

# Pollutants data from Environmental Protection Agency
URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# Read the Excel file and print
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
print("US air pollutant emission data:")
print(df)

# Show info
print("\nInformation about the DataFrame:")
df.info()

# print dtyes
coltypes = df.dtypes
print("\nColumn data types of the DataFrame:")
print(coltypes)

# Get last 3 columns
cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
print("\nDataFrame of last 3 years data:")
print(last3years)

# Get a series
data2021 = df["emissions21"]
print("\nSeries of 2021 data:")
print(data2021)

# Print unique pollutants
print("\nUnique pollutants:")
print(df["Pollutant"].unique())

# print mean emission
print("\nMean on the 2021 series:")
print(df["emissions21"].mean())

# Describe
print("\nBasic statistics about each column in the DataFrame:")
print(df.describe().T)

# Get CO only
df_CO = df[df["Pollutant"] == "CO"]
print("\nDataFrame of only CO pollutant:")
print(df_CO)

# Get CO and Highway only
df_CO_HW = df[(df["Pollutant"] == "CO")
              & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
print("\nDataFrame of only CO pollutant from Highway vehicles:")
print(df_CO_HW)

# Get DF of all CO
df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
print("\nDataFrame of only CO pollutant, keep only essential columns:")
print(df_all_co)

# Pivot
df_pivot = df_all_co.pivot_table(index="State",
                                 columns="Tier 1 Description",
                                 values="emissions21")
print("\nPivot table of state vs CO emission source:")
print(df_pivot)

# melt
df_melt = df_pivot.melt(value_name="emissions 2021",
                        var_name="Tier 1 Description",
                        ignore_index=False)
print("\nMelting the pivot table:")
print(df_melt)

# all three are the same
df_filled = df_pivot.fillna(0)
df_filled = df_pivot.where(df_pivot.notna(), 0)
df_filled = df_pivot.mask(df_pivot.isna(), 0)
print("\nFilled missing value as zero:")
print(df_filled)

# aggregation
df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
print("\nTotal CO emission by state:")
print(df_sum)

# group by
df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
print("\nComparing CO and SO2 emission:")
print(df_2021)

# join
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
print("\nComparing CO and SO2 emission:")
print(df_joined)

# merge
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"}) \
                                 .reset_index()
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"}) \
                                   .reset_index()
df_merged = df_co.merge(df_so2, on="State", how="outer")
print("\nComparing CO and SO2 emission:")
print(df_merged)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    minyear = sum_series[year_indices].astype(float).idxmin()
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
print("\nYears of minimum and maximum emissions:")
print(df_years)
