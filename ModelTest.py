from models.SpreadSIRModel import SpreadSIRModel
from models.BaseSIR import BaseSIRModel

Ds = lambda t: 1 - 10 ** (-t)
Di = lambda t: 1 - 3 ** (-t)
Dr = lambda t: 1 - 6 ** (-t)

time = 100  # Days
dt = 1  # Day
areaX = 1000
dx = 4
dy = 4
beta = .6  # Infectivity
gamma = .1  # Recovery

model = SpreadSIRModel(time, dt, beta, gamma, Ds, Di, Dr, dx, dy)
print('Loading image')
model.loadImage('grid.png')
print('Running Simulation')
model.runSimulation()
print('Done model')
