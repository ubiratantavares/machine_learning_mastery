import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.cloud import translate_v2 as translate
from tradutor.markdown import MarkdownHandler
from tradutor.extractor import TextExtractor


# Traduz texto utilizando a API Google Cloud Translation
class Translator:
    def __init__(self, credentials_path, target_language="pt"):
        self.credentials_path = credentials_path
        self.target_language = target_language
        self.translate_client = self._initialize_translate_client()

    def _initialize_translate_client(self):
        return translate.Client.from_service_account_json(self.credentials_path)

    def translate_text(self, text):
        if not text:
            raise ValueError("No text provided for translation.")
        
        # Quebra o texto em linhas
        lines = text.splitlines()
        translated_lines = []
        
        # Traduz cada linha individualmente
        for line in lines:
            if line.strip():  # Verifica se a linha não está vazia
                translated_line = self.translate_client.translate(line, target_language=self.target_language)
                translated_lines.append(translated_line["translatedText"])
            else:
                translated_lines.append("")  # Mantém a linha vazia para preservar a formatação

        # Junta as linhas traduzidas com quebras de linha
        return "\n".join(translated_lines)
    
# extrair, traduzir e salvar os textos
class WebpageTranslator:
    def __init__(self, url, credentials_path, target_language="pt"):
        self.url = url
        self.credentials_path = credentials_path
        self.target_language = target_language

        self.text_extractor = TextExtractor(url)
        self.translator = Translator(credentials_path, target_language)
        self.markdown_handler = MarkdownHandler()

    def process(self):
        # Step 1: Extract text from URL
        original_text = self.text_extractor.extract_text()
        self.markdown_handler.save_to_markdown(original_text, "original_text.md")
        
        # Step 2: Translate the text
        translated_text = self.translator.translate_text(original_text)
        self.markdown_handler.save_to_markdown(translated_text, "translated_text.md")
        
        print("Original text saved to 'original_text.md'.")
        print("Translated text saved to 'translated_text.md'.")
        print("Process completed successfully.")

    def read_markdown(self, file_path):
        return self.markdown_handler.read_from_markdown(file_path)
