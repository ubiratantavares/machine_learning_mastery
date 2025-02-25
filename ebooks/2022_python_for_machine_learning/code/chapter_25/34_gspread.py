import random

import gspread
from sklearn.datasets import fetch_openml

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Connect to Google Sheet
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)

# Clear all data
spreadsheet.clear()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet
spreadsheet.append_rows(rows)

# Check the number of rows and columns in the spreadsheet
print(spreadsheet.row_count, spreadsheet.col_count)

# Read a random row of data
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
row = random.randint(2, spreadsheet.row_count)
readrange = f"A{row}:{maxcol}{row}"
data = spreadsheet.get(readrange)
print(data)
