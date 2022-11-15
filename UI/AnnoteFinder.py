import matplotlib.pyplot as plt


class AnnoteFinder:
    def __init__(self, xdata, ydata, annotes, simulation, ax=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata)) / float(len(xdata))) / 2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata)) / float(len(ydata))) / 2
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []
        self.simulation = simulation

    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                for x, y, a in self.data:
                    if clickX - self.xtol < x < clickX + self.xtol and clickY - self.ytol < y < clickY + self.ytol:
                        dx, dy = x - clickX, y - clickY
                        annotes.append((dx * dx + dy * dy, x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.visitNode(annote)

    def visitNode(self, annote):
        self.simulation.drawInfectionPlot(annote)
