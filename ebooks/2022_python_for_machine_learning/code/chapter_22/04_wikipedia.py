import requests

URL = "https://en.wikipedia.org/static/images/project-logos/enwiki.png"
wikilogo = requests.get(URL)
if wikilogo.status_code == 200:
    with open("enwiki.png", "wb") as fp:
        fp.write(wikilogo.content)
