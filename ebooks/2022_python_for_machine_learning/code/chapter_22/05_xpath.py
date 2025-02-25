import requests
from bs4 import BeautifulSoup

# The numbers are lat-lon of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)
# Print the temperature
soup = BeautifulSoup(resp.text, "lxml")
elements = soup.select('span[data-testid="TemperatureValue"][class^="CurrentConditions"]')
print(elements[0].text)
