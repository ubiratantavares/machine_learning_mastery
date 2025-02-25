import requests
from lxml import etree
from bs4 import BeautifulSoup

# Reading temperature of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)

if resp.status_code == 200:
    # Using lxml
    dom = etree.HTML(resp.text)
    elements = dom.xpath("//span[@data-testid='TemperatureValue' and " \
                                   "contains(@class,'CurrentConditions')]")
    print(elements[0].text)

    # Using BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml")
    elements = soup.select('span[data-testid="TemperatureValue"]' \
                               '[class^="CurrentConditions"]')
    print(elements[0].text)
