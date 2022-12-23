import numpy as np
import cv2
import skvideo.io
from PyQt5.QtCore import QThread, pyqtSignal


class SpreadSIRModel(QThread):
    _signal = pyqtSignal(int)

    def __init__(self, Time, dt, beta, gamma, Ds, Di, Dr, dx, dy, population=1000000, grid=None):
        # Model related variables
        super(SpreadSIRModel, self).__init__()
        self.X = None
        self.Y = None
        self.population = population
        self.beta = beta
        self.gamma = gamma
        # Area and time related variables
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.Time = Time
        self.t = np.arange(0, Time, self.dt)
        # Dispersion related variables
        self.Ds = Ds
        self.Di = Di
        self.Dr = Dr
        # SIR Model variables
        self.grid = None
        self.Sprev = None
        self.Iprev = None
        self.Rprev = None
        self.Scur = None
        self.Icur = None
        self.Rcur = None
        self.partition = None
        self.path = None

    def loadImage(self, imagePath):
        # R - Recovered, G - Infected, B - Susceptible
        try:
            image = cv2.imread(imagePath)
            self.X, self.Y = image.shape[:2]
            self.grid = np.zeros((self.X, self.Y), dtype=np.longdouble)
            self.Sprev = np.zeros((self.X, self.Y), dtype=np.longdouble)
            self.Iprev = np.zeros((self.X, self.Y), dtype=np.longdouble)
            self.Rprev = np.zeros((self.X, self.Y), dtype=np.longdouble)
            self.grid = np.zeros((self.X, self.Y), dtype=np.longdouble)
            self.partition = self.population / (image[:, :, 0].sum() + image[:, :, 1].sum() + image[:, :, 2].sum())
            self.Sprev[:, :] = np.ceil(image[:, :, 0] * self.partition)
            self.Iprev[:, :] = np.ceil(image[:, :, 1] * self.partition)
            self.Rprev[:, :] = np.ceil(image[:, :, 2] * self.partition)
            self.grid = self.Sprev + self.Iprev + self.Rprev
            self.Scur = self.Sprev.copy()
            self.Icur = self.Iprev.copy()
            self.Rcur = self.Rprev.copy()
            return True
        except:
            return False

    def run(self):
        self.run2D()

    def run2D(self):
        import os
        img = np.zeros((self.grid.shape[0], self.grid.shape[1], 3))
        img[:, :, 0] = self.Rprev / self.partition
        img[:, :, 1] = self.Iprev / self.partition
        img[:, :, 2] = self.Sprev / self.partition
        writer = skvideo.io.FFmpegWriter(os.path.join(self.path, "SIRModel.mp4"))
        writer.writeFrame(img)
        for timeStep in range(self.Time):
            self._signal.emit(timeStep + 1)
            self.Sprev = self.Scur.copy()
            self.Iprev = self.Icur.copy()
            self.Rprev = self.Rcur.copy()
            S1 = self.Sprev[2:, 1:-1]  # S_{i+1, j}
            S_1 = self.Sprev[:-2, 1:-1]  # S_{i-1, j}
            Si1 = self.Sprev[1:-1, 2:]  # S_{i, j-1}
            Si_1 = self.Sprev[1:-1, :-2]  # S_{i, j+1}
            S = self.Sprev[1:-1, 1:-1]  # S_{i, k}

            I1 = self.Iprev[2:, 1:-1]  # I_{i+1, j}
            I_1 = self.Iprev[:-2, 1:-1]  # I_{i-1, j}
            Ii1 = self.Iprev[1:-1, 2:]  # I_{i, j-1}
            Ii_1 = self.Iprev[1:-1, :-2]  # I_{i, j+1}
            I = self.Iprev[1:-1, 1:-1]  # I_{i, k}

            R1 = self.Rprev[2:, 1:-1]  # R_{i+1, j}
            R_1 = self.Rprev[:-2, 1:-1]  # R_{i-1, j}
            Ri1 = self.Rprev[1:-1, 2:]  # R_{i, j-1}
            Ri_1 = self.Rprev[1:-1, :-2]  # R_{i, j+1}
            R = self.Rprev[1:-1, 1:-1]  # R_{i, k}
            self.Scur[1:-1, 1:-1] = self.Ds(self.dt * timeStep) * (
                    (Si_1 + Si1 - 2 * S) / self.dx ** 2 + (
                    S_1 + S1 - 2 * S) / self.dy ** 2) - self.beta * S * I + S
            self.Icur[1:-1, 1:-1] = self.Di(self.dt * timeStep) * ((Ii_1 + Ii1 - 2 * I) / self.dx ** 2 + (
                    I_1 + I1 - 2 * I) / self.dy ** 2) + self.beta * S * I - self.gamma * I + I
            self.Rcur[1:-1, 1:-1] = self.Dr(self.dt * timeStep) * (
                    (Ri_1 + Ri1 - 2 * R) / self.dx ** 2 + (R_1 + R1 - 2 * R) / self.dy ** 2) + self.gamma * I + R

            self.Scur[self.Scur < 0] = 0
            self.Icur[self.Icur < 0] = 0
            self.Rcur[self.Rcur < 0] = 0

            self.boundary2d()
            self.grid = self.Scur + self.Icur + self.Rcur
            img[:, :, 0] = self.Rcur / self.partition
            img[:, :, 1] = self.Icur / self.partition
            img[:, :, 2] = self.Scur / self.partition
            writer.writeFrame(img)

    def calculateDifference(self):
        print(f'Sus: {self.Scur.sum() - self.Sprev.sum()}, Inf: {self.Icur.sum() - self.Iprev.sum()}, '
              f'Rec: {self.Rcur.sum() - self.Rprev.sum()}')
        print(f'Diff between cur pop vs grid pop: {self.grid.sum() - self.population}')

    def boundary2d(self):
        self.Scur[0:self.grid.shape[0] - 1, 0] = self.Scur[0:self.grid.shape[0] - 1, 1]
        self.Icur[0:self.grid.shape[0] - 1, 0] = self.Icur[0:self.grid.shape[0] - 1, 1]
        self.Rcur[0:self.grid.shape[0] - 1, 0] = self.Rcur[0:self.grid.shape[0] - 1, 1]
        self.Scur[0:self.grid.shape[0] - 1, -1] = self.Scur[self.grid.shape[0] - 1, -2]
        self.Icur[0:self.grid.shape[0] - 1, -1] = self.Icur[self.grid.shape[0] - 1, -2]
        self.Rcur[0:self.grid.shape[0] - 1, -1] = self.Rcur[self.grid.shape[0] - 1, -2]

        self.Scur[0, 0:self.grid.shape[1] - 1] = self.Scur[0, 0:self.grid.shape[1] - 1]
        self.Icur[0, 0:self.grid.shape[1] - 1] = self.Icur[0, 0:self.grid.shape[1] - 1]
        self.Rcur[0, 0:self.grid.shape[1] - 1] = self.Rcur[0, 0:self.grid.shape[1] - 1]
        self.Scur[-1, 0:self.grid.shape[1] - 1] = self.Scur[-2, self.grid.shape[1] - 1]
        self.Icur[-1, 0:self.grid.shape[1] - 1] = self.Icur[-2, self.grid.shape[1] - 1]
        self.Rcur[-1, 0:self.grid.shape[1] - 1] = self.Rcur[-2, self.grid.shape[1] - 1]

    def setParameters(self, time, dt, dx, dy, beta, gamma, population):
        self.Time = time
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.beta = beta
        self.gamma = gamma
        self.population = population
