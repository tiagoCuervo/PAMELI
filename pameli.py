import copy
import random
import numpy as np
from consts import feasibleFuncApproximators, feasibleSearchAlgorithms
from pyDOE import lhs
import pygmo.core as pg
from pygmo import hypervolume
from utils import SurrogateProblem, rouletteSelection, balancedWeighting


class PAMELI(object):
    def __init__(self, problem, popSize, initSampleSize, mutProb, numCandidateSols, stochastic, adaptative):
        self.problem = problem
        self.popSize = popSize
        self.mutProb = mutProb
        self.numCandidateSols = numCandidateSols
        self.stochastic = stochastic
        self.adaptative = adaptative

        decSpaceDim = problem.decSpaceDim
        self.objSpaceDim = objSpaceDim = problem.objSpaceDim

        initDecSpaceSample = lhs(decSpaceDim, samples=initSampleSize)
        initDecSpaceScore = problem.evalFitness(initDecSpaceSample)
        # self.normalizationMean = np.mean(initDecSpaceScore, axis=0)
        # self.normalizationStdDev = np.std(initDecSpaceScore, axis=0)
        self.normalizationFactor = np.max(initDecSpaceScore, axis=0)
        initDecSpaceScore = initDecSpaceScore / self.normalizationFactor
        _, _, _, initParetoRanks = pg.fast_non_dominated_sorting(initDecSpaceScore)
        initWeights = balancedWeighting(initParetoRanks)
        self.dataset = {'decSpaceSamples': initDecSpaceSample,
                        'objSpaceImage': initDecSpaceScore,
                        'weights': initWeights}

        self.landscapeIdentifiers = []
        for i in range(popSize):
            functionApproximators = []
            for j in range(objSpaceDim):
                functionApproximators.append(np.random.choice(feasibleFuncApproximators)(decSpaceDim))
            surrogateModel = SurrogateModel(functionApproximators, self.normalizationFactor)
            self.landscapeIdentifiers.append(LandscapeIdentifier(surrogateModel,
                                                                 np.random.choice(feasibleSearchAlgorithms)))
        self.landscapeIdentifiers = np.array(self.landscapeIdentifiers)
        self.bestLandscapeIdentifier = None
        self.bestLandscapeIdentifierScore = None
        self.ItCounter = 0

    def sampleCandidateSolutions(self, estimatedParetoSets):
        candidateSols = []
        candidateIds = np.random.choice(self.popSize, size=self.numCandidateSols)
        if self.stochastic:
            means = []
            jointSet = []
            for solutionSet in estimatedParetoSets:
                means.append(np.mean(solutionSet, 0))
                jointSet = solutionSet if len(jointSet) == 0 else np.concatenate((jointSet, solutionSet), axis=0)
            covMatrix = np.cov(jointSet.T)
            for i in range(self.numCandidateSols):
                candidateSols.append(np.random.multivariate_normal(means[candidateIds[i]], covMatrix))
        else:
            for i in range(self.numCandidateSols):
                chosenSet = estimatedParetoSets[candidateIds[i]]
                candidateSols.append(chosenSet[np.random.choice(chosenSet.shape[0]), :])
        candidateSols = np.array(candidateSols)
        candidateSols[candidateSols < 0] = 0
        candidateSols[candidateSols > 1] = 1
        return candidateSols, candidateIds

    def evaluatePopulation(self, candidatesScore, candidatesId):
        refPoint = np.max(candidatesScore, 0) + 1e-12
        popFitness = np.zeros((self.popSize,))
        for i in range(self.popSize):
            hypervolumeIndicator = hypervolume(candidatesScore[candidatesId == i])
            popFitness[i] = hypervolumeIndicator.compute(refPoint)
        newBest = False
        bestIndividualIdx = int(np.argmax(popFitness))
        if self.ItCounter == 0:
            newBest = True
        else:
            if np.logical_and.reduce(np.max(self.bestLandscapeIdentifierScore, 0) < refPoint):
                hypervolumeIndicator = hypervolume(self.bestLandscapeIdentifierScore)
                bestIndividualUpdatedFitness = hypervolumeIndicator.compute(refPoint)
                if popFitness[bestIndividualIdx] > bestIndividualUpdatedFitness:
                    newBest = True
                else:
                    if self.adaptative:
                        self.landscapeIdentifiers = np.append(self.landscapeIdentifiers, self.bestLandscapeIdentifier)
                        popFitness = np.append(popFitness, bestIndividualUpdatedFitness)
            else:
                newBest = True
        if newBest:
            self.bestLandscapeIdentifierScore = candidatesScore[candidatesId == bestIndividualIdx]
            self.bestLandscapeIdentifier = self.landscapeIdentifiers[bestIndividualIdx]
        return popFitness

    def crossIndividuals(self, parent1, parent2):
        functionApproximators = []
        for surrogateGenIdx in range(self.objSpaceDim):
            if random.random() >= 0.5:
                functionApproximators.append(copy.deepcopy(parent1.surrogateModel.funcApproximators[surrogateGenIdx]))
            else:
                functionApproximators.append(copy.deepcopy(parent2.surrogateModel.funcApproximators[surrogateGenIdx]))
        surrogateModel = SurrogateModel(functionApproximators, self.normalizationFactor)
        if random.random() >= 0.5:
            searchAlgorithm = copy.deepcopy(parent1.searchAlgorithm)
        else:
            searchAlgorithm = copy.deepcopy(parent2.searchAlgorithm)
        return LandscapeIdentifier(surrogateModel, searchAlgorithm)

    # noinspection PyTypeChecker
    def updatePopulation(self, popFitness, verbose=False):
        parents = rouletteSelection(self.landscapeIdentifiers, popFitness, minNumOfParents=np.floor(self.popSize / 2))
        if verbose:
            print('----------- Genotype of selected parents -----------')
            for i in range(len(parents)):
                print('{}. '.format(i + 1) + parents[i].getSpecs())
            print('-----------------------------------------------------')

        self.landscapeIdentifiers = np.array([copy.deepcopy(self.bestLandscapeIdentifier)])
        for individualIdx in range(1, self.popSize):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            self.landscapeIdentifiers = np.append(self.landscapeIdentifiers, self.crossIndividuals(parent1, parent2))
            # self.landscapeIdentifiers[individualIdx] = self.crossIndividuals(parent1, parent2)
            self.landscapeIdentifiers[-1].mutate(self.mutProb)

    def iterate(self, verbose=False):
        if verbose:
            print('--------- Genotype of Landscape Identifiers ---------')
            for i in range(self.popSize):
                print('{}. '.format(i + 1) + self.landscapeIdentifiers[i].getSpecs())
            print('-----------------------------------------------------')
        estimatedParetoSets = []
        for idx, landscapeIdentifier in enumerate(self.landscapeIdentifiers):
            if verbose:
                print('Training surrogate model #{}'.format(idx + 1))
            landscapeIdentifier.surrogateModel.train(self.dataset)
            if verbose:
                print('Estimating Pareto Set for landscape identifier #{}'.format(idx + 1))
                estimatedParetoSets.append(
                    landscapeIdentifier.searchAlgorithm.searchForSolutions(
                        problem=landscapeIdentifier.surrogateModel.getSurrogateProblem(),
                        numCandidateSolutions=self.numCandidateSols))
        candidateSolutions, candidateIds = self.sampleCandidateSolutions(estimatedParetoSets)
        candidatesScore = self.problem.evalFitness(candidateSolutions) / self.normalizationFactor
        self.dataset['decSpaceSamples'] = np.concatenate((self.dataset['decSpaceSamples'], candidateSolutions), axis=0)
        self.dataset['objSpaceImage'] = np.concatenate((self.dataset['objSpaceImage'], candidatesScore), axis=0)
        _, _, _, sampleParetoRanks = pg.fast_non_dominated_sorting(self.dataset['objSpaceImage'])
        self.dataset['weights'] = balancedWeighting(sampleParetoRanks)
        popFitness = self.evaluatePopulation(candidatesScore, candidateIds)
        if self.adaptative:
            self.updatePopulation(popFitness, verbose)
        self.ItCounter += 1
        return self.dataset, self.bestLandscapeIdentifier


class LandscapeIdentifier:
    def __init__(self, surrogateModel, searchAlgorithm):
        self.surrogateModel = surrogateModel
        self.searchAlgorithm = searchAlgorithm

    def mutate(self, mutProb):
        for i in range(len(self.surrogateModel.funcApproximators)):
            if random.random() <= mutProb:
                # self.surrogateModel.funcApproximators[i].mutate()
                self.surrogateModel.funcApproximators[i] = np.random.choice(feasibleFuncApproximators)(
                    self.surrogateModel.funcApproximators[i].inputSpaceDim)
        if random.random() <= mutProb:
            self.searchAlgorithm = np.random.choice(feasibleSearchAlgorithms)

    def getSpecs(self):
        specs = ''
        for i in range(len(self.surrogateModel.funcApproximators)):
            specs = specs + '{}: '.format(i + 1) + self.surrogateModel.funcApproximators[i].name + '(' + \
                    self.surrogateModel.funcApproximators[i].getSpecs() + '),\t'
        specs = specs + '{}: '.format(len(self.surrogateModel.funcApproximators) + 1) + self.searchAlgorithm.name
        return specs


class SurrogateModel:
    def __init__(self, funcApproximators, normalizationFactor):
        self.funcApproximators = funcApproximators
        self.normalizationFactor = normalizationFactor

    def getSurrogateProblem(self):
        return SurrogateProblem(self.funcApproximators, self.normalizationFactor)

    # def surrogateProblem2(self):
    #     return SurrogateProblem2(self.funcApproximators).problem

    # def evalBatchFitness(self, batch):
    #     fitnessValues = []
    #     for i in range(len(self.funcApproximators)):
    #         ithObjectiveValue = self.funcApproximators[i].evalFunction(batch)
    #         if len(ithObjectiveValue.shape) == 1:
    #             ithObjectiveValue = np.expand_dims(ithObjectiveValue, 1)
    #         fitnessValues = np.concatenate((fitnessValues, ithObjectiveValue), axis=1) if i > 0 else ithObjectiveValue
    #     return fitnessValues

    def train(self, dataset):
        for i in range(len(self.funcApproximators)):
            print('  Training function approximator #{}'.format(i + 1))
            self.funcApproximators[i].train(dataset['decSpaceSamples'],
                                            dataset['objSpaceImage'][:, i],
                                            dataset['weights'])
