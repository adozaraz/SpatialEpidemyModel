from scipy.integrate import odeint


class BaseSIRModel:
    def __init__(self, SIRParams, time, population, beta, gamma):
        self.SIR = SIRParams # Вид данных - [S, I, R]
        self.time = time
        self.population = population
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def __derivativeModel(y, t, N, beta, gamma):
        S, I, R = y
        dSdT = - beta * S * I / N
        dIdT = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdT, dIdT, dRdt

    def calculateModel(self):
        modelData = odeint(BaseSIRModel.__derivativeModel, self.SIR, self.time, args=(self.population, self.beta, self.gamma))
        return modelData.T
