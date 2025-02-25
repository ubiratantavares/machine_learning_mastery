import pandas as pd

tables = pd.read_html("https://www.federalreserve.gov/releases/h15/")
print(tables)
