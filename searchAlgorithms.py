import pygmo.core as pg
# from pygmo.plotting import plot_non_dominated_fronts
# import matplotlib.pyplot as plt
# from platypus.algorithms import GDE3
import numpy as np
from pyDOE import lhs


class SearchAlgorithm(object):
    def __init__(self, numGens):
        self.algorithm = None
        self.numGens = numGens
        self.args = {
            'gen': self.numGens
        }

    def searchForSolutions(self, problem, numCandidateSolutions):
        problem = pg.problem(problem)
        population = pg.population(problem, size=numCandidateSolutions)
        algorithm = pg.algorithm(self.algorithm(**self.args))
        population = algorithm.evolve(population)
        return population.get_x()

    # def searchForSolutions2(self, problem, numCandidateSolutions):
    #     algo = GDE3(problem=problem, population_size=numCandidateSolutions)
    #     algo.run(1)
    #     # pf = []
    #     ps = []
    #     for solution in algo.result:
    #         # noinspection PyProtectedMember
    #         # pf.append(solution.objectives._data)
    #         ps.append(solution.variables)
    #     ps = np.array(ps)
    #     # pf = np.array(pf)
    #     return ps

    def mutate(self):
        pass


class NSGAII(SearchAlgorithm):
    def __init__(self, numGens):
        super().__init__(numGens)
        self.algorithm = pg.nsga2
        self.name = 'NSGAII'


class MOEAD(SearchAlgorithm):
    def __init__(self, numGens):
        super().__init__(numGens)
        self.algorithm = pg.moead
        self.args = {
            'gen': self.numGens,
            'weight_generation': 'low discrepancy'
        }
        self.name = 'MOEAD'


class MHACO(SearchAlgorithm):
    def __init__(self, numGens):
        super().__init__(numGens)
        self.algorithm = pg.maco
        self.name = 'MHACO'


class NSPSO(SearchAlgorithm):
    def __init__(self, numGens):
        super().__init__(numGens)
        self.algorithm = pg.nspso
        self.name = 'NSPSO'


class RandomSearch(SearchAlgorithm):
    def __init__(self, sampleSize):
        super().__init__(None)
        self.name = 'RandomSearch'
        self.sampleSize = sampleSize

    def searchForSolutions(self, problem, numCandidateSolutions):
        randomDecSpaceSample = lhs(len(problem.get_bounds()[0]), samples=self.sampleSize)
        sampleFitness = problem.evalBatchFitness(randomDecSpaceSample)
        fronts, _, _, _ = pg.fast_non_dominated_sorting(sampleFitness)
        nonDominatedSols = randomDecSpaceSample[fronts[0]]
        if nonDominatedSols.shape[0] > numCandidateSolutions:
            selectedIdxs = np.random.choice(np.arange(nonDominatedSols.shape[0]), size=numCandidateSolutions,
                                            replace=False)
            nonDominatedSols = nonDominatedSols[selectedIdxs]
        return nonDominatedSols
