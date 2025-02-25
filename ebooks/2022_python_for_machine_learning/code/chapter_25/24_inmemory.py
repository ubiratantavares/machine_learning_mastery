import random

import numpy as np
import openpyxl
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read data from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
header = list(dataset.columns)
rows = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Diabetes"
sheet.append(header)
for row in rows:
    sheet.append(row)
# Save
wb.save("MLM.xlsx")

# Read entire worksheet from the Excel file
wb = openpyxl.load_workbook("MLM.xlsx", read_only=True)
sheet = wb.active
X = []
y = []
for i, row in enumerate(sheet.rows):
    if i==0:
        continue # skip the header row
    rowdata = [cell.value for cell in row]
    X.append(rowdata[:-1])
    y.append(1 if rowdata[-1]=="tested_positive" else 0)
X, y = np.asarray(X), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(X, y, epochs=5)
