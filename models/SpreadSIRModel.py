import numpy as np
import cv2


def calculateGradient(lower, current, upper, h):
    return (lower - 2 * current + upper) / (h ** 2)


class SpreadSIRModel:
    def __init__(self, Time, dt, LengthX, dx, population, beta, gamma, Ds, Di, Dr, LengthY=None, dy=None,
                 LengthZ=None, dz=None):
        # Model related variables
        self.population = population
        self.beta = beta
        self.gamma = gamma
        # Area and time related variables
        self.dt = dt
        self.dx = dx
        self.Time = Time
        self.lengthX = LengthX
        self.x = np.arange(0, LengthX, self.dx)
        self.t = np.arange(0, Time, self.dt)
        self.shape = [self.t.shape[0], self.x.shape[0]]  # Structure: (T, X, [Y], [Z]) [] - not always present
        if not (LengthY is None and dy is None):
            self.lengthY = LengthY
            self.dy = dy
            self.y = np.arange(0, LengthY, self.dy)
            self.shape.append(self.y.shape[0])
            if not (LengthZ is None and dz is None):
                self.lengthZ = LengthZ
                self.dz = dz
                self.z = np.arange(0, LengthZ, self.dz)
                self.shape.append(self.z.shape[0])
        # Dispersion related variables
        self.Ds = Ds
        self.Di = Di
        self.Dr = Dr
        # SIR Model variables
        self.S = np.zeros(self.shape, dtype=float)
        self.I = np.zeros(self.shape, dtype=float)
        self.R = np.zeros(self.shape, dtype=float)
        self.sims = {2: self.run1D, 3: self.run2D, 4: self.run3D}
        self.drawers = {2: self.draw1D, 3: self.draw2D, 4: self.draw3D}

    def runSimulation(self):
        self.spreadPeople()
        print(self.S[0].sum() + self.I[0].sum() + self.R[0].sum())
        self.sims[len(self.shape)]()

    def spreadPeople(self):
        self.S[0] = self.population * 0.9 * np.random.dirichlet(np.ones(self.shape[1:]))
        self.I[0] = self.population * 0.1 * np.random.dirichlet(np.ones(self.shape[1:]))

    def run1D(self):
        for (i, t) in enumerate(self.t[1:], start=1):
            for (j, x) in enumerate(self.x[1:], start=1):
                if j >= self.x.shape[0] - 1:
                    continue
                self.S[i][j] = self.S[i - 1][j] + self.dS1D(i - 1, j) * self.dt
                if self.S[i][j] < 0:
                    self.S[i][j] = 0
                self.I[i][j] = self.I[i - 1][j] + self.dI1d(i - 1, j) * self.dt
                if self.I[i][j] < 0:
                    self.I[i][j] = 0
                self.R[i][j] = self.R[i - 1][j] + self.dR1d(i - 1, j) * self.dt
                if self.R[i][j] < 0:
                    self.R[i][j] = 0

            self.boundary1d(i - 1)

    def draw(self):
        self.drawers[len(self.shape)]()

    def draw1D(self):
        # R - Recovered, G - Infected, B - Susceptible
        print(self.I[-1].sum() + self.S[-1].sum() + self.R[-1].sum())
        print(self.population - (self.I[-1].sum() + self.S[-1].sum() + self.R[-1].sum()))
        Z = np.ndarray((self.shape[0], self.shape[1], 3))
        for step, tmp in enumerate(self.t, start=0):
            for coord, tmp in enumerate(self.x, start=0):
                colormap = {self.S[step][coord]: np.uint8((255, 0, 0)),
                            self.I[step][coord]: np.uint8((0, 255, 0)),
                            self.R[step][coord]: np.uint8((0, 0, 255))}
                Z[step][coord] = colormap[max(self.S[step][coord], self.I[step][coord], self.R[step][coord])]

        img = np.ndarray((self.shape[0] * 10, self.shape[1], 3))
        for step in range(0, self.shape[0] * 10, 10):
            for i in range(10):
                if step + i >= self.shape[0]:
                    break
                img[step + i] = Z[step]

        cv2.imshow("SIR", img)
        cv2.waitKey()

    def draw2D(self):
        pass

    def draw3D(self):
        pass

    def run2D(self):
        pass

    def run3D(self):
        pass

    def dS1D(self, i, j):
        # i - Time, j - X
        return self.Ds(self.dt) * (self.S[i][j - 1] + self.S[i][j + 1] - 2 * self.S[i][j]) / self.dx ** 2 - self.beta * \
            self.S[i][j] * self.I[i][j] / self.population

    def dI1d(self, i, j):
        return self.Di(self.dt) * (self.I[i][j - 1] + self.I[i][j + 1] - 2 * self.I[i][j]) / self.dx ** 2 + self.beta * \
            self.S[i][j] * self.I[i][j] / self.population - self.gamma * self.I[i][j]

    def dR1d(self, i, j):
        return self.Dr(self.dt) * (self.R[i][j - 1] + self.R[i][j + 1] - 2 * self.R[i][j]) / self.dx ** 2 + self.gamma * \
            self.I[i][j]

    def boundary1d(self, i):
        self.S[i + 1][0] = self.Ds(self.dt) * (2 * self.S[i][1] - 2 * self.S[i][0]) / self.dx ** 2 - self.beta * \
                           self.S[i][0] * self.I[i][0] / self.population
        if self.S[i][0] < 0:
            self.S[i][0] = 0
        self.S[i + 1][-1] = self.Ds(self.dt) * (2 * self.S[i][-2] - 2 * self.S[i][-1]) / self.dx ** 2 - self.beta * \
                            self.S[i][-1] * self.I[i][-1] / self.population
        if self.S[i][-1] < 0:
            self.S[i][-1] = 0

        self.I[i + 1][0] = self.Di(self.dt) * (2 * self.I[i][1] - 2 * self.I[i][0]) / self.dx ** 2 + self.beta * \
                           self.S[i][0] * self.I[i][0] / self.population - self.gamma * self.I[i][0]
        if self.I[i][0] < 0:
            self.I[i][0] = 0
        self.I[i + 1][-1] = self.Di(self.dt) * (2 * self.I[i][-2] - 2 * self.I[i][-1]) / self.dx ** 2 + self.beta * \
                            self.S[i][-1] * self.I[i][-1] / self.population - self.gamma * self.I[i][-1]
        if self.I[i][-1] < 0:
            self.I[i][-1] = 0

        self.R[i + 1][0] = self.Dr(self.dt) * (2 * self.R[i][1] - 2 * self.R[i][0]) / self.dx ** 2 + self.gamma * \
                           self.I[i][0]
        self.R[i + 1][-1] = self.Dr(self.dt) * (2 * self.R[i][-2] - 2 * self.R[i][-1]) / self.dx ** 2 + self.gamma * \
                            self.I[i][-1]
