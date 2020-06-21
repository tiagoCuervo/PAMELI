# import numpy as np
import argparse
import datetime
import os
import pickle

from problems import DTLZ, WFG
from pameli import PAMELI, pg
from utils import plotFrontDTLZProblem, plt

debugging = False

parser = argparse.ArgumentParser(description='PAMELI')
parser.add_argument('--problem', default='DTLZ2', help='Problem to solve (default: DTLZ2)')
parser.add_argument('--numExperiments', type=int, default=10, help='Number of experiments (default: 10)')
parser.add_argument('--numIters', type=int, default=10, help='Number of iterations (default: 10)')
parser.add_argument('--popSize', type=int, default=8,
                    help='Size of the population of Landscape Identifiers (default: 8)')
parser.add_argument('--initSampleSize', type=int, default=100, help='Initial dataset size (default: 100)')
parser.add_argument('--numCandidateSols', type=int, default=100,
                    help='Size of the set of promising solutions (default: 100)')
parser.add_argument('--mutProb', type=float, default=0.1,
                    help='Probability of mutation (default: 0.1)')
parser.add_argument('--verbose', default=True, help='Print log data to console (default: True)')
parser.add_argument('--deterministic', dest='stochastic', action='store_false',
                    help='Deterministic sampling')
parser.add_argument('--nonAdaptative', dest='adaptative', action='store_false',
                    help='Static population')
parser.set_defaults(adaptative=True)
parser.set_defaults(stochastic=True)

if __name__ == '__main__':
    args = parser.parse_args()
    problem = eval(args.problem[:-1])(problemId=int(args.problem[-1]))
    outputDir = './Results/{date:%d-%m-%Y_%H-%M}_'.format(date=datetime.datetime.now()) + problem.name
    if not os.path.exists(outputDir) and not debugging:
        os.makedirs(outputDir)
    for expIdx in range(args.numExperiments):
        expOutputDir = outputDir + '/exp{}'.format(expIdx + 1)
        if not os.path.exists(expOutputDir) and not debugging:
            os.makedirs(expOutputDir)
        algorithm = PAMELI(problem, args.popSize, args.initSampleSize, args.mutProb, args.numCandidateSols,
                           args.stochastic, args.adaptative)
        for iterIdx in range(args.numIters):
            print('----------- PAMELI, experiment #{}, iteration #{} -----------'.format(expIdx + 1, iterIdx + 1))
            archive, bestLandscapeIdentifier = algorithm.iterate(args.verbose)
            print('Estimating Pareto Set of best Landscape Identifier')
            # noinspection PyRedeclaration
            estimatedParetoSet = bestLandscapeIdentifier.searchAlgorithm.searchForSolutions(
                problem=bestLandscapeIdentifier.surrogateModel.getSurrogateProblem(),
                numCandidateSolutions=100)
            estimatedParetoFront = algorithm.problem.evalFitness(estimatedParetoSet)
            _, _, _, paretoRank = pg.fast_non_dominated_sorting(estimatedParetoFront)
            estimatedParetoSet = estimatedParetoSet[paretoRank == 0]
            _, _, _, paretoRank = pg.fast_non_dominated_sorting(archive['objSpaceImage'])
            archiveNonDominated = (archive['objSpaceImage'] * algorithm.normalizationFactor)[paretoRank == 0]
            if not debugging:
                # estimatedParetoFront = algorithm.problem.evalFitness(estimatedParetoSet)
                if args.problem[:-1] == 'DTLZ':
                    fig = plt.figure(1, figsize=(12, 6))
                    ax = fig.add_subplot(121, projection='3d')
                    plotFrontDTLZProblem(estimatedParetoFront, algorithm.problem.problem, ax=ax)
                    plotFrontDTLZProblem(archiveNonDominated, algorithm.problem.problem, ax=ax, marker='bo')
                    ax = fig.add_subplot(122, projection='3d')
                    ax.plot(archiveNonDominated[:, 0], archiveNonDominated[:, 1], archiveNonDominated[:, 2], 'bo')
                    ax.plot(estimatedParetoFront[:, 0], estimatedParetoFront[:, 1], estimatedParetoFront[:, 2], 'ro')
                    ax.view_init(azim=40)
                else:
                    fig = plt.figure(1, figsize=(12, 6))
                    plt.plot(archiveNonDominated[:, 0], archiveNonDominated[:, 1], 'bo')
                    plt.plot(estimatedParetoFront[:, 0], estimatedParetoFront[:, 1], 'ro')
                    # plotFrontDTLZProblem(estimatedParetoFront, algorithm.problem.problem, ax=ax)
                plt.savefig(expOutputDir + '/it#{}.png'.format(iterIdx + 1))
                plt.clf()
                with open(expOutputDir + '/it#{}.data'.format(iterIdx + 1), 'wb') as fp:
                    pickle.dump((archive, bestLandscapeIdentifier, estimatedParetoFront), fp)
