from models.BaseSIR import BaseSIRModel
import numpy as np


class MigrationSIR(BaseSIRModel):
    def simulationStep(self, i, *migration):
        # Структура миграции: (Theta, S, I)
        # Структура Data: (S, I)
        migrationData = np.array(list(map(lambda x: [x[0] * x[1], x[0] * x[2]], migration)))
        self.S[i] = self.S[i - 1] - self.beta * self.dt * self.I[i - 1] * self.S[i - 1] + migrationData[0]
        self.I[i] = self.I[i - 1] + \
                    self.dt * (self.beta * self.I[i - 1] * self.S[i - 1] - self.gamma * self.I[i - 1]) + migrationData[
                        1]
        self.population += migrationData[0] + migrationData[1]
        self.R[i] = self.population - self.S[i] - self.I[i]
