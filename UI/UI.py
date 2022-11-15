from UI.program import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow
from models import BaseSIR, MigrationSIR, Simulation


class ProgramWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.connectButtons()
        self.setupRadioButtons()
        self.setDefaultParameters()
        self.MigrationSIR.toggle()
        self.model = MigrationSIR.MigrationSIR

    def setDefaultParameters(self):
        self.startingPopulation.setText('1000')
        self.popDeviation.setText('100')
        self.beta.setText('0.2')
        self.gamma.setText('0.1')
        self.betaDeviation.setText('0')
        self.gammaDeviation.setText('0')
        self.time.setText('160')
        self.dt.setText('1')
        self.citiesNum.setText('4')

    def connectButtons(self):
        self.pushButton.clicked.connect(self.plotGraph)

    def setupRadioButtons(self):
        self.BaseSIR.toggled.connect(self.changeModel('Base'))
        self.MigrationSIR.toggled.connect(self.changeModel('Migration'))
        self.NewtonSIR.toggled.connect(self.changeModel('Newton'))

    def changeModel(self, model):
        def setBaseSIR():
            self.model = BaseSIR.BaseSIRModel

        def setMigrationSIR():
            self.model = MigrationSIR.MigrationSIR

        def setNewtonSIR():
            pass

        if model == 'Base':
            return setBaseSIR
        if model == 'Migration':
            return setMigrationSIR
        if model == 'Newton':
            return setNewtonSIR

    def plotGraph(self):
        citiesNum = int(self.citiesNum.text())
        N = int(self.startingPopulation.text())
        populationDeviation = int(self.popDeviation.text())
        beta, gamma = float(self.beta.text()), float(self.gamma.text())
        betaDeviation, gammaDeviation = float(self.betaDeviation.text()), float(self.gammaDeviation.text())
        t = int(self.time.text())  # Дни
        dt = int(self.dt.text())
        simulation = Simulation.Simulation(self.model,
                                           N,
                                           t,
                                           dt,
                                           beta,
                                           gamma,
                                           citiesNum,
                                           populationDeviation,
                                           betaDeviation,
                                           gammaDeviation)
        simulation.addWeights(0, 1)
        simulation.runSimulation()
        simulation.drawCities(True)
