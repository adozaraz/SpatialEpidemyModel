import numpy as np


class BaseSIRModel:
    def __init__(self, SIRParams, time, dt, population, beta, gamma):
        self.S = np.zeros(time, dtype=int)
        self.I = np.zeros(time, dtype=int)
        self.R = np.zeros(time, dtype=int)
        self.S[0] = SIRParams[0]
        self.I[0] = SIRParams[1]
        self.R[0] = SIRParams[2]
        self.time = time
        self.dt = dt
        self.population = population
        self.beta = beta
        self.gamma = gamma

    def simulationStep(self, i):
        self.S[i] = self.S[i - 1] - self.beta * self.dt * self.I[i - 1] * self.S[i - 1] / self.population
        self.I[i] = self.I[i - 1] + \
                    self.dt * (self.beta * self.I[i - 1] * self.S[i - 1] / self.population - self.gamma * self.I[i - 1])
        self.R[i] = self.population - self.S[i] - self.I[i]

    def runSimulation(self):
        for i in range(1, self.time):
            self.simulationStep(i)
