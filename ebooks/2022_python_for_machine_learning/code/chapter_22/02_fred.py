import io
import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE&cosd=2017-04-14&coed=2022-04-14"
resp = requests.get(URL)
if resp.status_code == 200:
   csvtext = resp.text
   csvbuffer = io.StringIO(csvtext)
   df = pd.read_csv(csvbuffer)
   print(df)
