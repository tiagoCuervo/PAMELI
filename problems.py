import numpy as np
from pygmo.core import problem, dtlz, wfg


class DTLZ(object):
    def __init__(self, problemId, objSpaceDim=3):
        self.name = 'DTLZ{}'.format(problemId)
        self.objSpaceDim = objSpaceDim
        self.decSpaceDim = self.getDecSpaceDim(problemId)
        self.problem = problem(dtlz(prob_id=problemId, dim=self.decSpaceDim, fdim=self.objSpaceDim))

    def getDecSpaceDim(self, problemId):
        if problemId == 1:
            return self.objSpaceDim + 4
        elif 2 <= problemId < 7:
            return self.objSpaceDim + 9
        elif problemId == 7:
            return self.objSpaceDim + 19
        else:
            raise NotImplementedError

    def evalFitness(self, population):
        return np.array([self.problem.fitness(individual) for individual in population.reshape(-1, self.decSpaceDim)])


class WFG(object):
    def __init__(self, problemId, objSpaceDim=2):
        self.name = 'WFG{}'.format(problemId)
        self.objSpaceDim = objSpaceDim
        self.decSpaceDim = self.getDecSpaceDim(problemId)
        self.problem = problem(wfg(prob_id=problemId, dim_dvs=self.decSpaceDim, dim_obj=self.objSpaceDim))

    def getDecSpaceDim(self, problemId):
        return 10

    def evalFitness(self, population):
        return np.array([self.problem.fitness(individual) for individual in population])
