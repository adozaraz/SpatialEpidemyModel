from models.SpreadSIRModel import SpreadSIRModel

Ds = lambda t: 1 - 10 ** (-5) * t
Di = lambda t: 1 + 10 ** (-3) * t
Dr = lambda t: 1 + 10 ** (-2) * t

model = SpreadSIRModel(1000, 1000, 2.5, 50, 1000, 0.3, 0.2, Ds, Di, Dr)

model.spreadPeopleRandomly()
model.runSimulation()
model.draw()
