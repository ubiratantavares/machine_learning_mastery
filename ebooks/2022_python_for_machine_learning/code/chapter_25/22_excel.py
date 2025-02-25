import pandas as pd
from sklearn.datasets import fetch_openml
import openpyxl

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
header = list(dataset.columns)
data = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.create_sheet("Diabetes")  # or wb.active for default sheet
sheet.append(header)
for row in data:
    sheet.append(row)
# Save
wb.save("MLM.xlsx")
