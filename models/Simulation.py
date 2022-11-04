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
        G = nx.DiGraph(nx.random_k_out_graph(self.citiesNum, 1, 0.5, self_loops=False))
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
        try:
            for i in range(1, self.time):
                for j in self.graph.nodes:
                    if self.modelType is MigrationSIR:
                        migrationData = self.getAdjacentData(j, i - 1)
                        self.graph.nodes[j]['model'].simulationStep(i, migrationData)
                    else:
                        self.graph.nodes[j]['model'].simulationStep(i)
        except Exception as e:
            print(f'Error has occurred during simulation. {e}')

        for i in self.graph.nodes:
            self.graph.add_node(i, data=(
                self.graph.nodes[i]['model'].S, self.graph.nodes[i]['model'].I, self.graph.nodes[i]['model'].R))

    def getAdjacentData(self, node, step):
        predecessors = self.graph.predecessors(node)
        data = []

        for edge in list(self.graph.edges(node, data=True)):
            node1, node2, tempData = edge
            data.append(
                [-tempData['weight'], self.graph.nodes[node1]['model'].S[step],
                 self.graph.nodes[node1]['model'].I[step]])

        for i in predecessors:
            for edge in list(self.graph.edges(i, data=True)):
                node1, node2, tempData = edge
                data.append(
                    [tempData['weight'], self.graph.nodes[i]['model'].S[step], self.graph.nodes[i]['model'].I[step]])

        return data

    def drawCities(self):
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

    def drawInfectionPlots(self):
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        fig = plt.figure()
        time = [i for i in range(self.time)]
        for i in self.graph.nodes:
            S, I, R = self.graph.nodes[i]['data']
            ax = fig.add_subplot(int(f'41{i + 1}'), facecolor='#dddddd', axisbelow=True)
            ax.plot(time, S, alpha=0.5, lw=2, label='Susceptible')
            ax.plot(time, I, alpha=0.5, lw=2, label='Infected')
            ax.plot(time, R, alpha=0.5, lw=2, label='Recovered')
            ax.set_xlabel('Time /days')
            ax.set_ylabel(f'Population in {i}')
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.text(0, 0,
                    f"Beta = {self.graph.nodes[i]['model'].beta:.2f}\nGamma: {self.graph.nodes[i]['model'].gamma:.2f}")
            ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
