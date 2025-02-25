from lxml import etree
import requests

# Read Yahoo home page
URL = "https://www.yahoo.com/"
resp = requests.get(URL)
dom = etree.HTML(resp.text)

# Print news headlines
elements = dom.xpath("//h3/a[u[@class='StretchedBox']]")
for elem in elements:
    print(etree.tostring(elem, method="text", encoding="unicode"))
