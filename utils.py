import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from platypus import Problem, Real
import scipy.stats


class RegressionDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# noinspection PyTypeChecker
class GaussianMembershipFunction(nn.Module):
    def __init__(self, numRules, inputSpaceDim):
        super().__init__()
        self.numRules = numRules
        self.inputSpaceDim = inputSpaceDim
        self.centers = Parameter(torch.randn(self.numRules * self.inputSpaceDim), requires_grad=True)
        self.widths = Parameter(torch.randn(self.numRules * self.inputSpaceDim), requires_grad=True)
        self.name = 'Gaussian'

    def forward(self, x):
        return torch.exp(-0.5 * (x.repeat(1, self.numRules) - self.centers) ** 2
                         / self.widths ** 2).view(-1, self.numRules, self.inputSpaceDim)


# noinspection PyTypeChecker
class SigmoidMembershipFunction(nn.Module):
    def __init__(self, numRules, inputSpaceDim):
        super().__init__()
        self.numRules = numRules
        self.inputSpaceDim = inputSpaceDim
        self.centers = Parameter(torch.randn(self.numRules * self.inputSpaceDim), requires_grad=True)
        self.widths = Parameter(torch.randn(self.numRules * self.inputSpaceDim), requires_grad=True)
        self.name = 'Sigmoid'

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.widths * (x.repeat(1, self.numRules) - self.centers))) \
            .view(-1, self.numRules, self.inputSpaceDim)


def evalSurrogates(surrogates, x):
    fitnessValues = []
    for i in range(len(surrogates)):
        ithObjectiveValue = surrogates[i].evalFunction(np.expand_dims(x, 0))
        if len(ithObjectiveValue.shape) == 1:
            ithObjectiveValue = np.expand_dims(ithObjectiveValue, 1)
        fitnessValues = np.concatenate((fitnessValues, ithObjectiveValue), axis=1) if i > 0 else ithObjectiveValue
        return fitnessValues


class SurrogateProblem:
    def __init__(self, surrogates, normalizationParams):
        self.surrogates = surrogates
        self.normalizationParams = normalizationParams

    def evalBatchFitness(self, batch):
        fitnessValues = []
        for i in range(len(self.surrogates)):
            ithObjectiveValue = self.surrogates[i].evalFunction(batch)
            fitnessValues = np.concatenate((fitnessValues, ithObjectiveValue.reshape(-1, 1)),
                                           axis=1) if i > 0 else ithObjectiveValue.reshape(-1, 1)
        return fitnessValues * self.normalizationParams

    def fitness(self, x):
        return self.evalBatchFitness(x).squeeze().tolist()

    def get_nobj(self):
        return len(self.surrogates)

    def get_bounds(self):
        decSpaceDim = self.surrogates[0].inputSpaceDim
        return [0.0] * decSpaceDim, [1.0] * decSpaceDim


# class SurrogateProblem2:
#     def __init__(self, surrogates):
#         self.surrogates = surrogates
#         self.problem = Problem(self.surrogates[0].inputSpaceDim, len(self.surrogates))
#         self.problem.types[:] = Real(0, 1)
#         self.problem.function = self.fitness
#         self.problem.name = 'SurrogateProblem2'
#
#     def fitness(self, x):
#         fitnessValues = []
#         for i in range(len(self.surrogates)):
#             ithObjectiveValue = self.surrogates[i].evalFunction(np.expand_dims(x, 0))
#             if len(ithObjectiveValue.shape) == 1:
#                 ithObjectiveValue = np.expand_dims(ithObjectiveValue, 1)
#             fitnessValues = np.concatenate((fitnessValues, ithObjectiveValue),
#                                            axis=1) if i > 0 else ithObjectiveValue
#         return fitnessValues.squeeze().tolist()
#
#     def get_nobj(self):
#         return len(self.surrogates)
#
#     def get_bounds(self):
#         decSpaceDim = self.surrogates[0].inputSpaceDim
#         return [0] * decSpaceDim, [1] * decSpaceDim


def computeConfidenceInterval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data, axis=0), scipy.stats.sem(data, axis=0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def plotFrontDTLZProblem(objSpaceVectors, problem, ax, az=40, marker='ro'):
    objSpaceDim = objSpaceVectors.shape[1]
    assert objSpaceDim == 3
    # Plot pareto front for dtlz 1
    if problem.get_name()[-1] in ["1"]:
        X, Y = np.meshgrid(np.linspace(0, 0.5, 100), np.linspace(0, 0.5, 100))
        Z = - X - Y + 0.5
        for i in range(100):
            for j in range(100):
                if X[i, j] < 0 or Y[i, j] < 0 or Z[i, j] < 0:
                    Z[i, j] = float('nan')
        ax.set_xlim(0, 1.)
        ax.set_ylim(0, 1.)
        ax.set_zlim(0, 1.)
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        plt.plot([0, 0.5], [0.5, 0], [0, 0])
    # Plot pareto fronts for DTLZ 2,3 & 4
    if problem.get_name()[-1] in ["2", "3", "4"]:
        thetas = np.linspace(0, (np.pi / 2.0), 30)
        gammas = np.linspace(0, (np.pi / 2.0), 30)

        x_frame = np.outer(np.cos(thetas), np.cos(gammas))
        y_frame = np.outer(np.cos(thetas), np.sin(gammas))
        z_frame = np.outer(np.sin(thetas), np.ones(np.size(gammas)))

        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.set_autoscalez_on(False)

        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.8)
        ax.set_zlim(0, 1.8)

        ax.plot_wireframe(x_frame, y_frame, z_frame)

    ax.plot(objSpaceVectors[:, 0], objSpaceVectors[:, 1], objSpaceVectors[:, 2], marker)
    ax.view_init(azim=az)
    return ax


def plotFrontDTLZProblem2(objSpaceVectors, problem, ax, az=40, marker='ro'):
    objSpaceDim = objSpaceVectors.shape[1]
    assert objSpaceDim == 3
    # Plot pareto front for dtlz 1
    if problem[-1] in ["1"]:
        X, Y = np.meshgrid(np.linspace(0, 0.5, 100), np.linspace(0, 0.5, 100))
        Z = - X - Y + 0.5
        for i in range(100):
            for j in range(100):
                if X[i, j] < 0 or Y[i, j] < 0 or Z[i, j] < 0:
                    Z[i, j] = float('nan')
        ax.set_xlim(0, 1.)
        ax.set_ylim(0, 1.)
        ax.set_zlim(0, 1.)
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        plt.plot([0, 0.5], [0.5, 0], [0, 0])
    # Plot pareto fronts for DTLZ 2,3 & 4
    if problem[-1] in ["2", "3", "4"]:
        thetas = np.linspace(0, (np.pi / 2.0), 30)
        gammas = np.linspace(0, (np.pi / 2.0), 30)

        x_frame = np.outer(np.cos(thetas), np.cos(gammas))
        y_frame = np.outer(np.cos(thetas), np.sin(gammas))
        z_frame = np.outer(np.sin(thetas), np.ones(np.size(gammas)))

        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.set_autoscalez_on(False)

        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.8)
        ax.set_zlim(0, 1.8)

        ax.plot_wireframe(x_frame, y_frame, z_frame)

    ax.plot(objSpaceVectors[:, 0], objSpaceVectors[:, 1], objSpaceVectors[:, 2], marker)
    ax.view_init(azim=az)
    return ax


def rouletteSelection(pop, fitness, minNumOfParents=2):
    normalizedFitness = np.zeros((fitness.shape[0],))
    for i in range(len(pop)):
        normalizedFitness[i] = fitness[i] / np.sum(fitness)
    selectionProbs = np.cumsum(np.sort(normalizedFitness))
    sortedIdxs = np.argsort(normalizedFitness)
    parentsIdxs = []
    while len(parentsIdxs) < minNumOfParents:
        chances = np.random.uniform(size=(selectionProbs.shape[0],))
        parentsIdxs = sortedIdxs[selectionProbs > chances]
    return pop[parentsIdxs]


def powerWeighting(paretoRanks, power=1):
    return 1 / (1 + paretoRanks) ** power


def symmetricWeighting(paretoRanks):
    return np.cos((paretoRanks * 2 * np.pi) / max(paretoRanks)) + 2


def balancedWeighting(paretoRanks):
    numOfRanks = int(max(paretoRanks) + 1)
    instancesCounter = [0.] * numOfRanks
    for rank in paretoRanks:
        instancesCounter[rank] += 1
    weightPerFront = [0.] * numOfRanks
    numSamples = len(paretoRanks)
    for i in range(numOfRanks):
        weightPerFront[i] = numSamples / instancesCounter[i]
    weights = [0.] * numSamples
    for idx, val in enumerate(paretoRanks):
        weights[idx] = weightPerFront[val]
    return weights
