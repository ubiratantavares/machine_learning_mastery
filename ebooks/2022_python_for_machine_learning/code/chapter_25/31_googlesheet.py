import random

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.datasets import fetch_openml

# Connect to Google Sheet
cred_file = "mlm-python.json"
scopes = ['https://www.googleapis.com/auth/spreadsheets']
cred = ServiceAccountCredentials.from_json_keyfile_name(cred_file, scopes)
service = build("sheets", "v4", http=cred.authorize(Http()))
sheet = service.spreadsheets()

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Delete everything on spreadsheet 0
body = {
    "requests": [{
        "deleteRange": {
            "range": {
                "sheetId": 0
            },
            "shiftDimension": "ROWS"
        }
    }]
}
action = sheet.batchUpdate(spreadsheetId=sheet_id, body=body)
action.execute()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet 0
maxcol = max(len(row) for row in rows)
maxcol = chr(ord("A") - 1 + maxcol)
action = sheet.values().append(
    spreadsheetId = sheet_id,
    body = {"values": rows},
    valueInputOption = "RAW",
    range = "Sheet1!A1:%s" % maxcol
)
action.execute()

# Check the sheets
sheet_properties = sheet.get(spreadsheetId=sheet_id).execute()["sheets"]
print(sheet_properties)

# Read a random row of data
maxrow = sheet_properties[0]["properties"]["gridProperties"]["rowCount"]
maxcol = sheet_properties[0]["properties"]["gridProperties"]["columnCount"]
maxcol = chr(ord("A") - 1 + maxcol)
row = random.randint(1, maxrow)
readrange = f"A{row}:{maxcol}{row}"
data = sheet.values().get(spreadsheetId=sheet_id, range=readrange).execute()
print(data)
