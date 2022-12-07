import numpy as np


def calculateGradient(lower, current, upper, h):
    return (lower - 2 * current + upper) / (h ** 2)


class SpreadSIRModel:
    def __init__(self, Time, nt, LengthX, nx, population, beta, gamma, Ds, Di, Dr, LengthY=None, ny=None,
                 LengthZ=None, nz=None):
        # Model related variables
        self.population = population
        self.beta = beta
        self.gamma = gamma
        # Area and time related variables
        self.ht = Time / nt
        self.hx = LengthX / nx
        self.x = np.linspace(-LengthX, LengthX, int(2 * LengthX / self.hx) + 1)
        self.t = np.linspace(0, Time, int(Time / self.ht) + 1)
        self.shape = [self.t.shape[0], self.x.shape[0]]  # Structure: (T, X, [Y], [Z]) [] - not always present
        if not (LengthY is None and ny is None):
            self.hy = LengthY / ny
            self.y = np.linspace(-LengthY, LengthY, int(2 * LengthY / self.hy) + 1)
            self.shape.append(self.y.shape[0])
            if not (LengthZ is None and nz is None):
                self.hz = LengthZ / nz
                self.z = np.linspace(-LengthZ, LengthZ, int(2 * LengthZ / self.hz) + 1)
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

    def runSimulation(self):
        self.sims[len(self.shape[1:])]()

    def run1DSimulation(self):
        for step in range(1, self.shape[0]):
            for i in range(1, self.shape[1]):
                gradientS = calculateGradient(self.S[step - 1][i - 1], self.S[step - 1][i], self.S[step - 1][i + 1],
                                              self.hx)
                gradientI = calculateGradient(self.I[step - 1][i - 1], self.I[step - 1][i], self.I[step - 1][i + 1],
                                              self.hx)
                gradientR = calculateGradient(self.R[step - 1][i - 1], self.R[step - 1][i], self.R[step - 1][i + 1],
                                              self.hx)

                self.S[step][i] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][i] * self.ht * (
                        1 - self.beta * self.I[step - 1][i])
                self.I[step][i] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][i] * self.ht * (
                        1 + self.beta * self.S[step - 1][i] - self.gamma)
                self.R[step][i] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][i] * self.ht + \
                                  self.gamma * self.I[step - 1][i] * self.ht
            # Conditions Sx=Ix=Rx=0
            gradientS = calculateGradient(self.S[step - 1][1], self.S[step - 1][0], self.S[step - 1][1],
                                          self.hx)
            gradientI = calculateGradient(self.I[step - 1][1], self.I[step - 1][0], self.I[step - 1][1],
                                          self.hx)
            gradientR = calculateGradient(self.R[step - 1][1], self.R[step - 1][0], self.R[step - 1][1],
                                          self.hx)
            self.S[step][0] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][0] * self.ht * (
                    1 - self.beta * self.I[step - 1][0])
            self.I[step][0] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][0] * self.ht * (
                    1 + self.beta * self.S[step - 1][0] - self.gamma)
            self.R[step][0] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][0] * self.ht + \
                              self.gamma * self.I[step - 1][0] * self.ht

            gradientS = calculateGradient(self.S[step - 1][-2], self.S[step - 1][-1], self.S[step - 1][-2],
                                          self.hx)
            gradientI = calculateGradient(self.I[step - 1][-2], self.I[step - 1][-1], self.I[step - 1][-2],
                                          self.hx)
            gradientR = calculateGradient(self.R[step - 1][-2], self.R[step - 1][-1], self.R[step - 1][-2],
                                          self.hx)
            self.S[step][-1] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][-1] * self.ht * (
                    1 - self.beta * self.I[step - 1][-1])
            self.I[step][-1] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][-1] * self.ht * (
                    1 + self.beta * self.S[step - 1][-1] - self.gamma)
            self.R[step][-1] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][-1] * self.ht + \
                               self.gamma * self.I[step - 1][-1] * self.ht

    def run2DSimulation(self):
        for step in range(1, self.shape[0]):
            for j in range(self.shape[2]):
                for i in range(1, self.shape[1]):
                    gradientS = calculateGradient(self.S[step - 1][j][i - 1], self.S[step - 1][j][i],
                                                  self.S[step - 1][j][i + 1],
                                                  self.hx) + \
                                calculateGradient(self.S[step - 1][j][i - 1],
                                                  self.S[step - 1][j][i],
                                                  self.S[step - 1][j][i + 1],
                                                  self.hy)

                    gradientI = calculateGradient(self.I[step - 1][j][i - 1], self.I[step - 1][j][i],
                                                  self.I[step - 1][j][i + 1],
                                                  self.hx) + \
                                calculateGradient(self.I[step - 1][j][i - 1],
                                                  self.I[step - 1][j][i],
                                                  self.I[step - 1][j][i + 1],
                                                  self.hy)

                    gradientR = calculateGradient(self.R[step - 1][j][i - 1], self.R[step - 1][j][i],
                                                  self.R[step - 1][j][i + 1],
                                                  self.hx) + \
                                calculateGradient(self.R[step - 1][j][i - 1],
                                                  self.R[step - 1][j][i],
                                                  self.R[step - 1][j][i + 1],
                                                  self.hy)

                    self.S[step][j][i] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][j][
                        i] * self.ht * (
                                                 1 - self.beta * self.I[step - 1][j][i])
                    self.I[step][j][i] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][j][
                        i] * self.ht * (
                                                 1 + self.beta * self.S[step - 1][j][i] - self.gamma)
                    self.R[step][j][i] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][j][
                        i] * self.ht + \
                                         self.gamma * self.I[step - 1][j][i] * self.ht

                gradientS = calculateGradient(self.S[step - 1][j][1], self.S[step - 1][j][0], self.S[step - 1][j][1],
                                              self.hx) + \
                            calculateGradient(self.S[step - 1][j][1], self.S[step - 1][j][0], self.S[step - 1][j][1],
                                              self.hy)

                gradientI = calculateGradient(self.I[step - 1][j][1], self.I[step - 1][j][0], self.I[step - 1][j][1],
                                              self.hx) + \
                            calculateGradient(self.I[step - 1][j][1], self.I[step - 1][j][0], self.I[step - 1][j][1],
                                              self.hy)

                gradientR = calculateGradient(self.R[step - 1][j][1], self.R[step - 1][j][0], self.R[step - 1][j][1],
                                              self.hx) + \
                            calculateGradient(self.R[step - 1][j][1], self.R[step - 1][j][0], self.R[step - 1][j][1],
                                              self.hy)

                self.S[step][j][0] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][j][
                    0] * self.ht * (
                                             1 - self.beta * self.I[step - 1][j][0])
                self.I[step][j][0] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][j][
                    0] * self.ht * (
                                             1 + self.beta * self.S[step - 1][j][0] - self.gamma)
                self.R[step][j][0] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][j][0] * self.ht + \
                                     self.gamma * self.I[step - 1][j][0] * self.ht

                gradientS = calculateGradient(self.S[step - 1][j][-2], self.S[step - 1][j][-1], self.S[step - 1][j][-2],
                                              self.hx) + \
                            calculateGradient(self.S[step - 1][j][-2],
                                              self.S[step - 1][j][-1],
                                              self.S[step - 1][j][-2],
                                              self.hy)

                gradientI = calculateGradient(self.I[step - 1][j][-2], self.I[step - 1][j][-1], self.I[step - 1][j][-2],
                                              self.hx) + \
                            calculateGradient(self.I[step - 1][j][-2], self.I[step - 1][j][-1], self.I[step - 1][j][-2],
                                              self.hy)

                gradientR = calculateGradient(self.R[step - 1][j][-2], self.R[step - 1][j][-1], self.R[step - 1][j][-2],
                                              self.hx) + \
                            calculateGradient(self.R[step - 1][j][-2], self.R[step - 1][j][-1], self.R[step - 1][j][-2],
                                              self.hy)

                self.S[step][j][-1] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][j][
                    -1] * self.ht * (
                                              1 - self.beta * self.I[step - 1][j][-1])
                self.I[step][j][-1] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][j][
                    -1] * self.ht * (
                                              1 + self.beta * self.S[step - 1][j][-1] - self.gamma)
                self.R[step][j][-1] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][j][
                    -1] * self.ht + \
                                      self.gamma * self.I[step - 1][j][-1] * self.ht

    def run3DSimulation(self):
        for step in range(1, self.shape[0]):
            for k in range(self.shape[3]):
                for j in range(self.shape[2]):
                    for i in range(self.shape[1]):
                        gradientS = calculateGradient(self.S[step - 1][k][j][i - 1], self.S[step - 1][k][j][i],
                                                      self.S[step - 1][k][j][i + 1],
                                                      self.hx) + \
                                    calculateGradient(self.S[step - 1][k][j][i - 1],
                                                      self.S[step - 1][k][j][i],
                                                      self.S[step - 1][k][j][i + 1],
                                                      self.hy) + \
                                    calculateGradient(self.S[step - 1][k][j][i - 1],
                                                      self.S[step - 1][k][j][i],
                                                      self.S[step - 1][k][j][i + 1],
                                                      self.hz)

                        gradientI = calculateGradient(self.I[step - 1][k][j][i - 1], self.I[step - 1][k][j][i],
                                                      self.I[step - 1][k][j][i + 1],
                                                      self.hx) + \
                                    calculateGradient(self.I[step - 1][k][j][i - 1],
                                                      self.I[step - 1][k][j][i],
                                                      self.I[step - 1][k][j][i + 1],
                                                      self.hy) + \
                                    calculateGradient(self.I[step - 1][k][j][i - 1],
                                                      self.I[step - 1][k][j][i],
                                                      self.I[step - 1][k][j][i + 1],
                                                      self.hz)

                        gradientR = calculateGradient(self.R[step - 1][k][j][i - 1], self.R[step - 1][k][j][i],
                                                      self.R[step - 1][k][j][i + 1],
                                                      self.hx) + \
                                    calculateGradient(self.R[step - 1][k][j][i - 1],
                                                      self.R[step - 1][k][j][i],
                                                      self.R[step - 1][k][j][i + 1],
                                                      self.hy) + \
                                    calculateGradient(self.R[step - 1][k][j][i - 1],
                                                      self.R[step - 1][k][j][i],
                                                      self.R[step - 1][k][j][i + 1],
                                                      self.hz)

                        self.S[step][k][j][i] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][k][j][
                            i] * self.ht * (1 - self.beta * self.I[step - 1][k][j][i])
                        self.I[step][k][j][i] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][k][j][
                            i] * self.ht * (1 + self.beta * self.S[step - 1][k][j][i] - self.gamma)
                        self.R[step][k][j][i] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][k][j][
                            i] * self.ht + self.gamma * self.I[step - 1][k][j][i] * self.ht

                    gradientS = calculateGradient(self.S[step - 1][k][j][1], self.S[step - 1][k][j][0],
                                                  self.S[step - 1][k][j][1],
                                                  self.hx) + \
                                calculateGradient(self.S[step - 1][k][j][1], self.S[step - 1][k][j][0],
                                                  self.S[step - 1][k][j][1],
                                                  self.hy) + \
                                calculateGradient(self.S[step - 1][k][j][1], self.S[step - 1][k][j][0],
                                                  self.S[step - 1][k][j][1],
                                                  self.hz)

                    gradientI = calculateGradient(self.I[step - 1][k][j][1], self.I[step - 1][k][j][0],
                                                  self.I[step - 1][k][j][1],
                                                  self.hx) + \
                                calculateGradient(self.I[step - 1][k][j][1], self.I[step - 1][k][j][0],
                                                  self.I[step - 1][k][j][1],
                                                  self.hy) + \
                                calculateGradient(self.I[step - 1][k][j][1], self.I[step - 1][k][j][0],
                                                  self.I[step - 1][k][j][1],
                                                  self.hz)

                    gradientR = calculateGradient(self.R[step - 1][k][j][1], self.R[step - 1][k][j][0],
                                                  self.R[step - 1][k][j][1],
                                                  self.hx) + \
                                calculateGradient(self.R[step - 1][k][j][1], self.R[step - 1][k][j][0],
                                                  self.R[step - 1][k][j][1],
                                                  self.hy) + \
                                calculateGradient(self.R[step - 1][k][j][1], self.R[step - 1][k][j][0],
                                                  self.R[step - 1][k][j][1],
                                                  self.hz)

                    self.S[step][j][0] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][j][
                        0] * self.ht * (
                                                 1 - self.beta * self.I[step - 1][j][0])
                    self.I[step][j][0] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][j][
                        0] * self.ht * (
                                                 1 + self.beta * self.S[step - 1][j][0] - self.gamma)
                    self.R[step][j][0] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][j][
                        0] * self.ht + \
                                         self.gamma * self.I[step - 1][j][0] * self.ht

                    gradientS = calculateGradient(self.S[step - 1][k][j][-2], self.S[step - 1][k][j][-1],
                                                  self.S[step - 1][k][j][-2],
                                                  self.hx) + \
                                calculateGradient(self.S[step - 1][k][j][-2],
                                                  self.S[step - 1][k][j][-1],
                                                  self.S[step - 1][k][j][-2],
                                                  self.hy) + \
                                calculateGradient(self.S[step - 1][k][j][-2],
                                                  self.S[step - 1][k][j][-1],
                                                  self.S[step - 1][k][j][-2],
                                                  self.hz)
                    gradientI = calculateGradient(self.I[step - 1][k][j][-2], self.I[step - 1][k][j][-1],
                                                  self.I[step - 1][k][j][-2],
                                                  self.hx) + \
                                calculateGradient(self.I[step - 1][k][j][-2], self.I[step - 1][k][j][-1],
                                                  self.I[step - 1][k][j][-2],
                                                  self.hy) + \
                                calculateGradient(self.I[step - 1][k][j][-2], self.I[step - 1][k][j][-1],
                                                  self.I[step - 1][k][j][-2],
                                                  self.hz)
                    gradientR = calculateGradient(self.R[step - 1][k][j][-2], self.R[step - 1][k][j][-1],
                                                  self.R[step - 1][k][j][-2],
                                                  self.hx) + \
                                calculateGradient(self.R[step - 1][k][j][-2], self.R[step - 1][k][j][-1],
                                                  self.R[step - 1][k][j][-2],
                                                  self.hy) + \
                                calculateGradient(self.R[step - 1][k][j][-2], self.R[step - 1][k][j][-1],
                                                  self.R[step - 1][k][j][-2],
                                                  self.hz)
                    self.S[step][k][j][-1] = self.Ds(step * self.ht) * gradientS * self.ht - self.S[step - 1][k][j][
                        -1] * self.ht * (
                                                     1 - self.beta * self.I[step - 1][k][j][-1])
                    self.I[step][k][j][-1] = self.Di(step * self.ht) * gradientI * self.ht + self.I[step - 1][k][j][
                        -1] * self.ht * (
                                                     1 + self.beta * self.S[step - 1][k][j][-1] - self.gamma)
                    self.R[step][k][j][-1] = self.Dr(step * self.ht) * gradientR * self.ht + self.R[step - 1][k][j][
                        -1] * self.ht + \
                                             self.gamma * self.I[step - 1][k][j][-1] * self.ht

    def spreadPeopleRandomly(self):
        susceptible = int(self.population * 0.9)
        infected = int(self.population * 0.1)
        self.S[0] = np.random.dirichlet(np.ones(self.shape[1:]), size=susceptible)
        self.I[0] = np.random.dirichlet(np.ones(self.shape[1:]), size=infected)
