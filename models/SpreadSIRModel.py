import numpy as np


class SpreadSIRModel:
    def __init__(self, Time, nt, Area, stepArea, population, beta, gamma, Ds, Di, Dr):
        self.Time = Time
        self.ht = Time / nt
        self.Area = Area
        self.hx = tuple(map(lambda x, nx: x / nx, zip(Area, stepArea)))
        self.population = population
        self.beta = beta
        self.gamma = gamma
        self.Ds = Ds
        self.Di = Di
        self.Dr = Dr
        self.x = np.linspace(-Length, Length, int(2 * Length / self.hx) + 1)
        self.t = np.linspace(0, Time, int(Time / self.ht) + 1)
        self.S = np.zeros((self.x.shape[0], self.t.shape[0]))
        self.I = np.zeros((self.x.shape[0], self.t.shape[0]))
        self.K = np.zeros((self.x.shape[0], self.t.shape[0]))

    def runSimulation(self):
        for i in range(self.t.shape[0] + 1):
            pass

    def spreadPeopleRandomly(self):
        randomPeople = np.random.dirichlet(np.ones(self.x.shape[0] - 2), size=1)
        self.S[0] = self.population * randomPeople
        self.S[0][0] =
