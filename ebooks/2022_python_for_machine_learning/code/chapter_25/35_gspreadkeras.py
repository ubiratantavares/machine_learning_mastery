import random

import numpy as np
import gspread
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Read the entire spreadsheet, except header
maxrow = spreadsheet.row_count
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
data = spreadsheet.get(f"A2:{maxcol}{maxrow}")
X = [row[:-1] for row in data]
y = [1 if row[-1]=="tested_positive" else 0 for row in data]
X, y = np.asarray(X).astype(float), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(X, y, epochs=5)
