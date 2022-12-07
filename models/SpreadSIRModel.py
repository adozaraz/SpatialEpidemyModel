import numpy as np


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
        self.shape = [self.t.shape[0], self.x.shape[0]]
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
        self.S = np.zeros(self.shape)
        self.I = np.zeros(self.shape)
        self.K = np.zeros(self.shape)

    def runSimulation(self):
        pass

    def spreadPeopleRandomly(self):
        randomPeople = np.random.dirichlet(np.ones(self.shape[1:]), size=1)
        pass
