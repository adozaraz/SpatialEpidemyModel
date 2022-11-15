from models.BaseSIR import BaseSIRModel
import numpy as np


class MigrationSIR(BaseSIRModel):
    def __init__(self, SIRParams, time, dt, population, beta, gamma):
        super().__init__(SIRParams, time, dt, population, beta, gamma)
        self.N = np.zeros(time, dtype=int)
        self.N[0] = self.population

    def simulationStep(self, i, *migration):
        import copy
        # Структура миграции: (Theta, S, I)
        # Структура Data: (S, I)
        migrationCopy = copy.copy(*migration)
        migrationData = np.sum(np.array(list(map(lambda x: [x[0] * x[1], x[0] * x[2]], migrationCopy))), axis=0)
        self.S[i] = self.S[i - 1] - (self.beta * self.dt * self.I[i - 1] * self.S[i - 1]) / self.population + \
                    migrationData[0]
        self.I[i] = self.I[i - 1] + self.dt * ((self.beta * self.I[i - 1] * self.S[i - 1]) / self.population -
                                               self.gamma * self.I[i - 1]) + migrationData[1]
        self.population += migrationData[0] + migrationData[1]
        self.R[i] = self.population - self.S[i] - self.I[i]
        self.N[i] = self.population
