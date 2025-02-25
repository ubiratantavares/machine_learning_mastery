import requests
from lxml import etree

# The numbers are lat-lon of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)
# Create DOM from HTML text
dom = etree.HTML(resp.text)
# Search for the temperature element and get the content
elements = dom.xpath("//span[@data-testid='TemperatureValue' and contains(@class,'CurrentConditions')]")
print(elements[0].text)
