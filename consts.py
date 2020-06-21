import torch
from funcApproximators import ANN, FIS, SVM, KNNRegressor, DecTreeRegressor
from utils import GaussianMembershipFunction, SigmoidMembershipFunction
from searchAlgorithms import NSGAII, MOEAD, MHACO, NSPSO, RandomSearch

# feasibleFuncApproximators = [ANN, FIS, SVM, KNNRegressor, DecTreeRegressor]
feasibleFuncApproximators = [ANN, FIS, SVM]
# ------------- Global params --------------
feasibleBatchSizes = [16]
# --------------- ANN params ---------------
# feasibleLearnRates = [1e-2, 1e-3, 1e-4, 1e-5]
feasibleLearnRates = [1e-2, 1e-3]
feasibleNumLayers = [1, 2, 3, 4]
# feasibleNumLayers = [1]
feasiblehiddenDims = [8, 16, 32, 64, 128, 256, 512, 1024]
# feasibleActivationFunctions = [torch.tanh, torch.relu, torch.sigmoid]
feasibleActivationFunctions = [torch.relu]
maxEpochs = 1000
# maxEpochs = 1
# --------------- FIS params ---------------
feasibleNumRules = [8, 16, 32, 64, 128]
# feasibleMembershipFunctions = [GaussianMembershipFunction, SigmoidMembershipFunction]
feasibleMembershipFunctions = [GaussianMembershipFunction]
# --------------- SVM params ---------------
# feasibleKernels = ['linear', 'poly', 'rbf', 'sigmoid']
feasibleKernels = ['rbf']
feasibleDegrees = [2, 3, 4, 5, 6, 7]
# --------------- KNN params ---------------
feasibleNumNeighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
feasibleWeights = ['distance', 'uniform']
# ------------- DecTree params -------------
# feasibleMaxDepths = [2, 4, 6, 8, None]
feasibleSplitters = ['best']
# ------------- Search params --------------
feasibleSearchPopSize = (64, 256)
numGens = 800
# numGens = 1
# feasibleSearchAlgorithms = [NSGAII(numGens=numGens), MOEAD(numGens=numGens), RandomSearch(sampleSize=10000)]
feasibleSearchAlgorithms = [NSGAII(numGens=numGens), MOEAD(numGens=numGens)]
