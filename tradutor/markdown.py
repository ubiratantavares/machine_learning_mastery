
# salvar e ler o conteúdo em arquivos markdown
class MarkdownHandler:

    @staticmethod
    def save_to_markdown(text, file_path):
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(text)
        print("Salvo com sucesso.")

    @staticmethod
    def read_from_markdown(file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read()
