from models.SpreadSIRModel import SpreadSIRModel

Ds = lambda t: 1 - 10 ** (-t)
Di = lambda t: 1 - 12 ** (-t)
Dr = lambda t: 1 - 11 ** (-t)

time = 365  # Days
dt = 1  # Day
dx = 4
dy = 4
beta = .8  # Infectivity
gamma = .2  # Recovery
population = 100

model = SpreadSIRModel(time, dt, beta, gamma, Ds, Di, Dr, dx, dy, population)
print('Loading image')
model.loadImage('BasicGrids/grid4.png')
print('Running Simulation')
model.runSimulation()
print('Done model')
