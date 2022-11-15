from PyQt5.QtWidgets import QWidget, QSizePolicy, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

matplotlib.use('QT5Agg')


class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('auto')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        self.plots = []

    def setLabels(self, xLabel, yLabel):
        self.ax.set_xlabel(xLabel)
        self.ax.set_ylabel(yLabel)

    def addPlot(self, x, y, label):
        self.plots.append(self.ax.plot(x, y, label=label))

    def clearPlots(self):
        self.ax.cla()
        legend = self.ax.get_legend()
        if legend is None:
            return
        legend.remove()


class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

    def setLabels(self, xLabel, yLabel):
        self.canvas.setLabels(xLabel, yLabel)

    def addPlot(self, x, y, label):
        self.canvas.addPlot(x, y, label)

    def draw(self):
        self.canvas.ax.legend(loc='lower right')
        self.canvas.draw()

    def clearPlot(self):
        self.canvas.clearPlots()
