from UI.program import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QErrorMessage
from models.SpreadSIRModel import SpreadSIRModel

Ds = lambda t: 1 - 10 ** (-t)
Di = lambda t: 1 - 12 ** (-t)
Dr = lambda t: 1 - 11 ** (-t)


class ProgramWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QMainWindow.__init__(self)
        self.boardChosen = False
        self.savePathChosen = False
        self.setupUi(self)
        self.connectButtons()
        self.setDefaultParameters()
        self.model = SpreadSIRModel(100, 1, .2, .1, Ds, Di, Dr, 2, 2, 1000000)
        self.model._signal.connect(self.signalAccept)
        self.simulationProgress.setVisible(False)
        self.pushButton.setEnabled(False)

    def setDefaultParameters(self):
        self.time.setText('100')
        self.dt.setText('1')
        self.betaDeviation.setText('0.2')
        self.gammaDeviation.setText('0.1')
        self.dx.setText('2')
        self.dy.setText('2')
        self.population.setText('1000000')

    def connectButtons(self):
        self.pushButton.clicked.connect(self.simulate)
        self.photoChoose.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.chooseSavePath)

    def loadImage(self):
        self.imagePath = QFileDialog.getOpenFileName(self, "Открыть файл", '')
        self.imagePath = self.imagePath[0]
        if self.imagePath != '':
            if self.model.loadImage(self.imagePath):
                self.boardChosen = True
                if self.savePathChosen and self.boardChosen:
                    self.pushButton.setEnabled(True)
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Information)
                msgBox.setText("Поле успешно загружено")
                msgBox.setWindowTitle("Оповещение")
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec()
            else:
                self.boardChosen = False
                self.pushButton.setEnabled(False)
                message = QErrorMessage()
                message.showMessage("Изображение невозможно загрузить")
                message.exec()

    def chooseSavePath(self):
        dirPath = str(QFileDialog.getExistingDirectory(None, "Выберите папку для сохранения"))
        if dirPath != '':
            self.model.path = dirPath
            self.savePathChosen = True
            if self.savePathChosen and self.boardChosen:
                self.pushButton.setEnabled(True)
        else:
            self.pushButton.setEnabled(False)
            self.savePathChosen = False

    def simulate(self):
        beta, gamma = float(self.betaDeviation.text()), float(self.gammaDeviation.text())
        t = int(self.time.text())  # Дни
        dt = int(self.dt.text())
        dx = int(self.dx.text())
        dy = int(self.dy.text())
        population = int(self.population.text())
        self.model.setParameters(t, dt, dx, dy, beta, gamma, population)
        self.pushButton.setEnabled(False)
        self.simulationProgress.setValue(0)
        self.simulationProgress.setVisible(True)
        self.model.start()

    def signalAccept(self, msg):
        value = int(msg) / int(self.time.text())
        self.simulationProgress.setValue(value * 100)
        print(value)
        if int(msg) == int(self.time.text()):
            self.simulationProgress.setVisible(False)
            self.simulationProgress.setValue(0)
            self.pushButton.setEnabled(True)
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Видео сохранено в папке с приложением под названием 'SIRModel.mp4'")
            msgBox.setWindowTitle("Оповещение")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
