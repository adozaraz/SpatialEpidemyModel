import random

import networkx as nx
import numpy as np
from models.MigrationSIR import MigrationSIR


class Simulation:
    def __init__(self, modelType, population, time, timeStep, beta, gamma, citiesNum,
                 populationDeviation=100, betaDeviation=0, gammaDeviation=0):
        self.population = population
        self.time = time
        self.dt = timeStep  # В днях
        self.beta = beta
        self.gamma = gamma
        self.citiesNum = citiesNum
        self.populationDeviation = populationDeviation
        self.betaDeviation = betaDeviation
        self.gammaDeviation = gammaDeviation
        self.modelType = modelType
        self.graph = self.generateGraphWithConnections()

    def generateGraph(self):
        return nx.gnm_random_graph(self.citiesNum, 0)

    def generateGraphWithConnections(self):
        import random
        G = nx.DiGraph(nx.random_k_out_graph(self.citiesNum, 1))
        for node in G.nodes:
            n = self.population + random.randrange(-self.populationDeviation, self.populationDeviation)
            mBeta = self.beta + random.uniform(-self.betaDeviation, self.betaDeviation) \
                if self.betaDeviation != 0 else self.beta
            mGamma = self.gamma + random.uniform(-self.gammaDeviation, self.gammaDeviation) \
                if self.gammaDeviation != 0 else self.gamma
            SIR = [0, int(n * (random.randrange(5, 10) / 100)), 0]
            SIR[0] = n - SIR[1]
            G.add_node(node, model=self.modelType(SIR, self.time, self.dt, n, mBeta, mGamma))
        return G

    def addWeights(self, minWeight, maxWeight):
        if maxWeight < minWeight:
            raise ValueError('MaxWeight cannot be more that MinWeight')
        if minWeight < 0. or maxWeight > 1.:
            raise ValueError('Weights should be from 0 to 1')
        for node in self.graph.nodes:
            if len(self.graph.edges(node)):
                for in_node, out_node in self.graph.edges(node):
                    self.graph.add_edge(in_node, out_node, weight=random.uniform(minWeight, maxWeight))

    def runSimulation(self):
        for i in range(self.time):
            for j in self.graph.nodes:
                if self.modelType is MigrationSIR:
                    migrationData = self.getAdjacentData(j, i)
                    self.graph.nodes[j]['model'].calculateStep(i, migrationData)
                else:
                    self.graph.nodes[j]['model'].calculateStep(i)

        for i in self.graph.nodes:
            self.graph.add_node(i, data=(
                self.graph.nodes[i]['model'].S, self.graph.nodes[i]['model'].I, self.graph.nodes[i]['model'].R))

    def getAdjacentData(self, node, step):
        successors = self.graph.successors(node)
        predecessors = self.graph.predecessors(node)
        data = []
        for i in successors:
            node1, node2, tempData = self.graph.edges(i, data=True)
            data.append([tempData['theta'], self.graph.nodes[i]['model'].S[step], self.graph.nodes[i]['model'].I[step]])

        for i in predecessors:
            node1, node2, tempData = self.graph.edges(i, data=True)
            data.append(
                [-tempData['theta'], self.graph.nodes[i]['model'].S[step], self.graph.nodes[i]['model'].I[step]])

        return data
