from models.SpreadSIRModel import SpreadSIRModel
from models.BaseSIR import BaseSIRModel

Ds = lambda t: 1 - 10 ** (-5) * t
Di = lambda t: 1 + 10 ** (-3) * t
Dr = lambda t: 1 + 10 ** (-2) * t

time = 100  # Days
dt = 1  # Day
areaX = 500
dx = 2
population = 10000000
beta = 0.2  # Infectivity
gamma = 0.1  # Recovery

model = SpreadSIRModel(time, dt, areaX, dx, population, beta, gamma, Ds, Di, Dr)

model.runSimulation()
model.draw()
print('Done model')
