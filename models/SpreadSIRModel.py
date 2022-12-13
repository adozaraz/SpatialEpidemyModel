import numpy as np
import cv2
import skvideo.io


class SpreadSIRModel:
    def __init__(self, Time, dt, beta, gamma, Ds, Di, Dr, dx, dy, population=None, grid=None):
        # Model related variables
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

    def loadImage(self, imagePath):
        # R - Recovered, G - Infected, B - Susceptible
        image = cv2.imread(imagePath)
        self.X, self.Y = image.shape[:2]
        self.grid = np.zeros((self.X, self.Y), dtype=np.float64)
        self.Sprev = np.zeros((self.X, self.Y), dtype=np.float64)
        self.Iprev = np.zeros((self.X, self.Y), dtype=np.float64)
        self.Rprev = np.zeros((self.X, self.Y), dtype=np.float64)
        self.Sprev[:, :] = image[:, :, 0] / 255
        self.Iprev[:, :] = image[:, :, 1] / 255
        self.Rprev[:, :] = image[:, :, 2] / 255
        # for i in range(self.X):
        #     for j in range(self.Y):
        #         if np.array_equal(image[i, j], [0, 0, 255]):
        #             self.grid[i, j] = 1
        #             self.Rprev[i, j] = 1
        #         elif np.array_equal(image[i, j], [0, 255, 0]):
        #             self.grid[i, j] = 1
        #             self.Iprev[i, j] = 1
        #         elif np.array_equal(image[i, j], [255, 0, 0]):
        #             self.grid[i, j] = 1
        #             self.Sprev[i, j] = 1
        self.grid = self.Sprev + self.Iprev + self.Rprev
        self.population = np.sum(self.grid)
        self.Scur = self.Sprev.copy()
        self.Icur = self.Iprev.copy()
        self.Rcur = self.Rprev.copy()

    def runSimulation(self):
        self.run2D()

    def run2D(self):
        img = np.zeros((self.grid.shape[0], self.grid.shape[1], 3))
        frameWidth = self.grid.shape[0]
        frameHeight = self.grid.shape[1]
        img = img.reshape((frameWidth, frameHeight, 3))
        frameSize = (frameWidth, frameHeight)
        fps = 20
        mask = self.grid != 0
        # img[mask, 0] = np.uint8(self.Sprev[mask] / self.grid[mask] * 255)
        # img[mask, 1] = np.uint8(self.Iprev[mask] / self.grid[mask] * 255)
        # img[mask, 2] = np.uint8(self.Rprev[mask] / self.grid[mask] * 255)
        img[mask, 1] = np.uint8(self.Iprev[mask] / self.grid[mask] * 255)
        img[mask, 0] = np.uint8(self.Rprev[mask] / self.grid[mask] * 255)
        img[mask, 2] = np.uint8(self.Sprev[mask] / self.grid[mask] * 255)
        writer = skvideo.io.FFmpegWriter("SIRModel.mp4")
        writer.writeFrame(img)
        cv2.imwrite(f'data/SIR{0}.jpg', img)
        for timeStep in range(self.Time):
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
            # for row in range(1, self.X - 1):
            #     for col in range(1, self.Y - 1):
            #         self.Scur[row, col] = self.dS2D(row, col, timeStep) * self.dt + self.Sprev[row, col]
            #         if self.Scur[row, col] < 0:
            #             self.Scur[row, col] = 0
            #         self.Icur[row, col] = self.dI2d(row, col, timeStep) * self.dt + self.Iprev[row, col]
            #         if self.Icur[row, col] < 0:
            #             self.Icur[row, col] = 0
            #         self.Rcur[row, col] = self.dR2d(row, col, timeStep) * self.dt + self.Rprev[row, col]
            #         if self.Rcur[row, col] < 0:
            #             self.Rcur[row, col] = 0

            self.boundary2d()
            self.grid = self.Scur + self.Icur + self.Rcur
            mask = self.grid != 0
            # img[mask, 0] = np.uint8(self.Scur[mask] / self.grid[mask] * 255)
            # img[mask, 1] = np.uint8(self.Rcur[mask] / self.grid[mask] * 255)
            # img[mask, 2] = np.uint8(self.Icur[mask] / self.grid[mask] * 255)
            img[mask, 1] = np.uint8(self.Icur[mask] / self.grid[mask] * 255)
            img[mask, 0] = np.uint8(self.Rcur[mask] / self.grid[mask] * 255)
            img[mask, 2] = np.uint8(self.Scur[mask] / self.grid[mask] * 255)
            # cv2.imwrite(f'data/SIR{timeStep}.jpg', img)
            writer.writeFrame(img)

    def dS2D(self, row, col, timeStep):
        return self.Ds(self.dt * timeStep) * (
                (self.Sprev[row, col + 1] + self.Sprev[row, col - 1] - 2 * self.Sprev[row, col]) / self.dx ** 2 +
                (self.Sprev[row + 1, col] + self.Sprev[row - 1, col] - 2 * self.Sprev[
                    row, col]) / self.dy ** 2) - self.beta * self.Sprev[row, col] * self.Iprev[
            row, col] / self.population

    def dI2d(self, row, col, timeStep):
        return self.Di(self.dt * timeStep) * (
                (self.Iprev[row, col + 1] + self.Iprev[row, col - 1] - 2 * self.Iprev[row, col]) / self.dx ** 2 +
                (self.Iprev[row + 1, col] + self.Iprev[row - 1, col] - 2 * self.Iprev[
                    row, col]) / self.dy ** 2) + self.beta * self.Sprev[row, col] * self.Iprev[
            row, col] / self.population - self.gamma * self.Iprev[row, col]

    def dR2d(self, row, col, timeStep):
        return self.Dr(self.dt * timeStep) * (
                (self.Rprev[row, col + 1] + self.Rprev[row, col - 1] - 2 * self.Rprev[row, col]) / self.dx ** 2 +
                (self.Rprev[row + 1, col] + self.Rprev[row - 1, col] - 2 * self.Rprev[
                    row, col]) / self.dy ** 2) + self.gamma * self.Iprev[row, col]

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
