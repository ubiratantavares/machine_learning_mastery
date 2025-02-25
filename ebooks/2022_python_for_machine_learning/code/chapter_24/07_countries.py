import pandas_datareader.wb as wb

countries = wb.get_countries()
print(countries)
