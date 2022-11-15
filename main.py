from UI.UI import ProgramWindow
from PyQt5.QtWidgets import QApplication
import sys

if __name__ in "__main__":
    app = QApplication(sys.argv)

    window = ProgramWindow()
    window.show()

    app.exec()
