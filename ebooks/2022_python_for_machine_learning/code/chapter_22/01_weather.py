import requests

# The numbers are lat-lon of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)
print(resp.status_code)
print(resp.text)
