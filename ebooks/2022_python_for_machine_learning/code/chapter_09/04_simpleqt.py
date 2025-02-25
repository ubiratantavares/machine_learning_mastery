import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow

class Frame(QMainWindow):
        def __init__(self):
                super().__init__()
                self.initUI()
        def initUI(self):
                self.setWindowTitle("Simple title")
                self.resize(800,600)

def main():
        app = QApplication(sys.argv)
        frame = Frame()
        frame.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
        main()
