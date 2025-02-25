import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tradutor.markdown import MarkdownHandler
from tradutor.translator import Translator

def exibir(text):
    lines = text.splitlines()
    for line in lines:
        print(line)
    return text

def main():
    file_path_original = input("Digite o nome do arquivo markdown para tradução: ")    
    original_text = MarkdownHandler.read_from_markdown(file_path_original)    
    translator = Translator("/home/bira/github/mlm/machine_learning_mastery/tradutor/graphic-charter-419118-54d28b3239b0.json")
    translated_text = translator.translate_text(original_text)
    file_path_translated = file_path_original.replace(".md", "_translated.md")
    MarkdownHandler.save_to_markdown(translated_text, file_path_translated)

if __name__ == "__main__":
    main()