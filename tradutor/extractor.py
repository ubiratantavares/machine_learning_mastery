import requests
from bs4 import BeautifulSoup

# extrair texto de uma página web
class TextExtractor:
    
    def __init__(self, url):
        self.url = url
        self.extracted_text = ""

    def extract_text(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.stripped_strings
        self.extracted_text = "\n\n".join(texts)
        return self.extracted_text
