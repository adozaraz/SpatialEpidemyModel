import numpy as np
import matplotlib.pyplot as plt


def calculateGradient(lower, current, upper, h):
    return (lower - 2 * current + upper) / (h ** 2)


class SpreadSIRModel:
    def __init__(self, Time, GridSize, LengthX, I, population, beta, gamma, Ds, Di, Dr, LengthY=None, J=None,
                 LengthZ=None, K=None):
        # Model related variables
        self.population = population
        self.beta = beta
        self.gamma = gamma
        # Area and time related variables
        self.ht = Time / GridSize
        self.hx = 2 * LengthX / I
        self.x = np.arange(-LengthX, LengthX, self.hx)
        self.t = np.arange(0, Time, self.ht)
        self.shape = [self.t.shape[0], self.x.shape[0]]  # Structure: (T, X, [Y], [Z]) [] - not always present
        if not (LengthY is None and J is None):
            self.hy = 2 * LengthY / J
            self.y = np.arange(-LengthY, LengthY, self.hy)
            self.shape.append(self.y.shape[0])
            if not (LengthZ is None and K is None):
                self.hz = 2 * LengthZ / K
                self.z = np.arange(-LengthZ, LengthZ, self.hz)
                self.shape.append(self.z.shape[0])
        # Dispersion related variables
        self.Ds = Ds
        self.Di = Di
        self.Dr = Dr
        # SIR Model variables
        self.S = np.zeros(self.shape, dtype=np.float32)
        self.I = np.zeros(self.shape, dtype=np.float32)
        self.R = np.zeros(self.shape, dtype=np.float32)
        self.sims = {1: self.run1DSimulation, 2: self.run2DSimulation, 3: self.run3DSimulation}
        self.drawings = {1: self.draw1DSim, 2: self.draw2DSim, 3: self.draw3DSim}

    def runSimulation(self):
        self.sims[len(self.shape[1:])]()

    def calculateS(self, lowerS, currentS, upperS, currentI, h, step):
        # h structure: [t, x, y, z]
        gradient = 0
        for i in h[1:]:
            gradient += calculateGradient(lowerS, currentS, upperS, i)

        return self.Ds(step * h[0]) * gradient * h[0] + currentS * h[0] * (1 - currentI ** 2)

    def calculateI(self, lowerI, currentI, upperI, currentS, h, step):
        gradient = 0
        for i in h[1:]:
            gradient += calculateGradient(lowerI, currentI, upperI, i)

        return self.Di(step * h[0]) * gradient * h[0] + currentI * h[0] * (1 + currentI * currentS - self.gamma)

    def calculateR(self, lowerR, currentR, upperR, currentI, h, step):
        gradient = 0
        for i in h[1:]:
            gradient += calculateGradient(lowerR, currentR, upperR, i)

        return self.Dr(step * h[0]) * gradient * h[0] + currentR * h[0] + self.gamma * currentI * h[0]

    def run1DSimulation(self):
        for step in range(1, self.shape[0]):
            for i in range(1, self.shape[1] - 1):
                self.S[step][i] = self.calculateS(self.S[step - 1][i - 1], self.S[step - 1][i], self.S[step - 1][i + 1],
                                                  self.I[step - 1][i], [self.ht, self.hx], step)
                self.I[step][i] = self.calculateI(self.I[step - 1][i - 1], self.I[step - 1][i], self.I[step - 1][i + 1],
                                                  self.S[step - 1][i], [self.ht, self.hx], step)
                self.R[step][i] = self.calculateR(self.R[step - 1][i - 1], self.R[step - 1][i], self.R[step - 1][i + 1],
                                                  self.I[step - 1][i], [self.ht, self.hx], step)

            # Conditions Sx=Ix=Rx=0
            self.S[step][0] = self.calculateS(self.S[step - 1][1], self.S[step - 1][0], self.S[step - 1][1],
                                              self.I[step - 1][0], [self.ht, self.hx], step)
            self.I[step][0] = self.calculateI(self.I[step - 1][1], self.I[step - 1][0], self.I[step - 1][1],
                                              self.S[step - 1][0], [self.ht, self.hx], step)
            self.R[step][0] = self.calculateR(self.R[step - 1][1], self.R[step - 1][0], self.R[step - 1][1],
                                              self.I[step - 1][0], [self.ht, self.hx], step)
            self.S[step][-1] = self.calculateS(self.S[step - 1][-2], self.S[step - 1][-1], self.S[step - 1][-2],
                                               self.I[step - 1][-1], [self.ht, self.hx], step)
            self.I[step][-1] = self.calculateI(self.I[step - 1][-2], self.I[step - 1][-1], self.I[step - 1][-2],
                                               self.S[step - 1][-1], [self.ht, self.hx], step)
            self.R[step][-1] = self.calculateR(self.R[step - 1][-2], self.R[step - 1][-1], self.R[step - 1][-2],
                                               self.I[step - 1][-1], [self.ht, self.hx], step)

            if self.S[step].sum() + self.I[step].sum() + self.R[step].sum() != 1:
                raise ValueError(f'SIR condition is not met. Step: {step}')

    def run2DSimulation(self):
        for step in range(1, self.shape[0]):
            for j in range(self.shape[2]):
                for i in range(1, self.shape[1] - 1):
                    self.S[step][i] = self.calculateS(self.S[step - 1][j][i - 1], self.S[step - 1][j][i],
                                                      self.S[step - 1][j][i + 1],
                                                      self.I[step - 1][j][i], [self.ht, self.hx, self.hy], step)
                    self.I[step][i] = self.calculateI(self.I[step - 1][j][i - 1], self.I[step - 1][j][i],
                                                      self.I[step - 1][j][i + 1],
                                                      self.S[step - 1][j][i], [self.ht, self.hx, self.hy], step)
                    self.R[step][i] = self.calculateR(self.R[step - 1][j][i - 1], self.R[step - 1][j][i],
                                                      self.R[step - 1][j][i + 1],
                                                      self.I[step - 1][j][i], [self.ht, self.hx, self.hy], step)

                    # Conditions Sx=Ix=Rx=0
                self.S[step][0] = self.calculateS(self.S[step - 1][j][1], self.S[step - 1][j][0],
                                                  self.S[step - 1][j][1],
                                                  self.I[step - 1][j][0], [self.ht, self.hx, self.hy], step)
                self.I[step][0] = self.calculateI(self.I[step - 1][j][1], self.I[step - 1][j][0],
                                                  self.I[step - 1][j][1],
                                                  self.S[step - 1][j][0], [self.ht, self.hx, self.hy], step)
                self.R[step][0] = self.calculateR(self.R[step - 1][j][1], self.R[step - 1][j][0],
                                                  self.R[step - 1][j][1],
                                                  self.I[step - 1][j][0], [self.ht, self.hx, self.hy], step)
                self.S[step][-1] = self.calculateS(self.S[step - 1][j][-2], self.S[step - 1][j][-1],
                                                   self.S[step - 1][j][-2],
                                                   self.I[step - 1][j][-1], [self.ht, self.hx, self.hy], step)
                self.I[step][-1] = self.calculateI(self.I[step - 1][j][-2], self.I[step - 1][j][-1],
                                                   self.I[step - 1][j][-2],
                                                   self.S[step - 1][j][-1], [self.ht, self.hx, self.hy], step)
                self.R[step][-1] = self.calculateR(self.R[step - 1][j][-2], self.R[step - 1][j][-1],
                                                   self.R[step - 1][j][-2],
                                                   self.I[step - 1][j][-1], [self.ht, self.hx, self.hy], step)

    def run3DSimulation(self):
        for step in range(1, self.shape[0]):
            for k in range(self.shape[3]):
                for j in range(self.shape[2]):
                    for i in range(self.shape[1] - 1):
                        self.S[step][i] = self.calculateS(self.S[step - 1][k][j][i - 1], self.S[step - 1][k][j][i],
                                                          self.S[step - 1][k][j][i + 1],
                                                          self.I[step - 1][k][j][i],
                                                          [self.ht, self.hx, self.hy, self.hz], step)
                        self.I[step][i] = self.calculateI(self.I[step - 1][k][j][i - 1], self.I[step - 1][k][j][i],
                                                          self.I[step - 1][k][j][i + 1],
                                                          self.S[step - 1][k][j][i],
                                                          [self.ht, self.hx, self.hy, self.hz], step)
                        self.R[step][i] = self.calculateR(self.R[step - 1][k][j][i - 1], self.R[step - 1][k][j][i],
                                                          self.R[step - 1][k][j][i + 1],
                                                          self.I[step - 1][k][j][i],
                                                          [self.ht, self.hx, self.hy, self.hz], step)

                        # Conditions Sx=Ix=Rx=0
                    self.S[step][0] = self.calculateS(self.S[step - 1][k][j][1], self.S[step - 1][k][j][0],
                                                      self.S[step - 1][k][j][1],
                                                      self.I[step - 1][k][j][0], [self.ht, self.hx, self.hy, self.hz],
                                                      step)
                    self.I[step][0] = self.calculateI(self.I[step - 1][k][j][1], self.I[step - 1][k][j][0],
                                                      self.I[step - 1][k][j][1],
                                                      self.S[step - 1][k][j][0], [self.ht, self.hx, self.hy, self.hz],
                                                      step)
                    self.R[step][0] = self.calculateR(self.R[step - 1][k][j][1], self.R[step - 1][k][j][0],
                                                      self.R[step - 1][k][j][1],
                                                      self.I[step - 1][k][j][0], [self.ht, self.hx, self.hy, self.hz],
                                                      step)
                    self.S[step][-1] = self.calculateS(self.S[step - 1][k][j][-2], self.S[step - 1][k][j][-1],
                                                       self.S[step - 1][k][j][-2],
                                                       self.I[step - 1][k][j][-1], [self.ht, self.hx, self.hy, self.hz],
                                                       step)
                    self.I[step][-1] = self.calculateI(self.I[step - 1][k][j][-2], self.I[step - 1][k][j][-1],
                                                       self.I[step - 1][k][j][-2],
                                                       self.S[step - 1][k][j][-1], [self.ht, self.hx, self.hy, self.hz],
                                                       step)
                    self.R[step][-1] = self.calculateR(self.R[step - 1][k][j][-2], self.R[step - 1][k][j][-1],
                                                       self.R[step - 1][k][j][-2],
                                                       self.I[step - 1][k][j][-1], [self.ht, self.hx, self.hy, self.hz],
                                                       step)

    def spreadPeopleRandomly(self):
        self.S[0] = 0.9 * np.random.dirichlet(np.ones(self.shape[1:]))
        self.I[0] = 0.1 * np.random.dirichlet(np.ones(self.shape[1:]))

    def draw(self):
        self.drawings[len(self.shape[1:])]()

    def draw1DSim(self):
        plt.plot(self.x, self.S[-1], label='Susceptible')
        plt.plot(self.x, self.I[-1], label='Infected')
        plt.plot(self.x, self.R[-1], label='Recovered')
        plt.legend()
        plt.show()

        plt.plot(self.t, self.S, label='Susceptible')
        plt.plot(self.t, self.I, label='Infected')
        plt.plot(self.t, self.R, label='Recovered')
        plt.legend()
        plt.show()

        fig, ax = plt.subplots()
        levelsS = np.linspace(self.S.min(initial=-999999), self.S.max(initial=0), 7)
        levelsI = np.linspace(self.I.min(initial=-999999), self.I.max(initial=0), 7)
        levelsR = np.linspace(self.R.min(initial=-999999), self.R.max(initial=0), 7)
        ax.plot(self.x, self.t, 'o', markersize=2)
        ax.tricontourf(self.t, self.x, self.S, levels=levelsS)
        ax.tricontourf(self.t, self.x, self.I, levels=levelsI)
        ax.tricontourf(self.t, self.x, self.R, levels=levelsR)
        plt.show()

    def draw2DSim(self):
        pass

    def draw3DSim(self):
        pass
