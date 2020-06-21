import random
import consts
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import RegressionDataset


class FunctionApproximator(object):
    def __init__(self, inputSpaceDim):
        self.inputSpaceDim = inputSpaceDim

    def genRandomSpecs(self):
        raise NotImplementedError

    def getSpecs(self):
        raise NotImplementedError

    def train(self, inputs, targets, weights=None):
        raise NotImplementedError

    def evalFunction(self, x):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError


# noinspection PyUnresolvedReferences,PyArgumentList
class ANN(FunctionApproximator):
    def __init__(self, inputSpaceDim):
        super().__init__(inputSpaceDim)
        self.specs = specs = self.genRandomSpecs()
        self.model = self.buildModel(inputSpaceDim, specs)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=specs['learnRate'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.batchSize = specs['batchSize']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = 'ANN'
        self.bestLoss = np.inf
        self.wait = 0
        self.patience = 10
        self.minDelta = 0

    def buildModel(self, inputSpaceDim, specs):
        return MLP(inputSpaceDim, 1, specs)

    def getSpecs(self):
        specs = '[' + str(self.specs['layerSizes']) + '], ['
        for i in range(len(self.specs['activationFunctions'])):
            if isinstance(self.specs['activationFunctions'][i], str):
                specs = specs + self.specs['activationFunctions'][i]
            else:
                specs = specs + self.specs['activationFunctions'][i].__name__
            if i < len(self.specs['activationFunctions']) - 1:
                specs = specs + ', '
            else:
                specs = specs + ']'
        specs = specs + ', ' + str(self.specs['learnRate'])
        return specs

    def genRandomSpecs(self):
        numLayers = np.random.choice(consts.feasibleNumLayers)
        batchSize = np.random.choice(consts.feasibleBatchSizes)
        learnRate = np.random.choice(consts.feasibleLearnRates)
        hiddenSize = np.random.choice(consts.feasiblehiddenDims)
        layerSizes = []
        # useBias = np.random.choice([True, False])
        useBias = True
        activationFunctions = []
        for i in range(numLayers):
            layerSizes.append(hiddenSize)
            activationFunctions.append(np.random.choice(consts.feasibleActivationFunctions))
        # activationFunctions.append(np.random.choice(consts.feasibleActivationFunctions + ['Linear']))
        # activationFunctions.append(np.random.choice(['Linear', torch.relu]))
        activationFunctions.append(torch.relu)
        # activationFunctions.append('Linear')
        specs = {'layerSizes': layerSizes,
                 'useBias': useBias,
                 'activationFunctions': activationFunctions,
                 'batchSize': batchSize,
                 'learnRate': learnRate}
        return specs

    def train(self, inputs, targets, weights=None):
        self.wait = 0
        self.bestLoss = np.inf
        inputs = Variable(torch.from_numpy(inputs.astype(np.float32)), requires_grad=False)
        targets = torch.from_numpy(targets.astype(np.float32))
        trainData = RegressionDataset(inputs, targets)
        if weights is not None:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            trainLoader = DataLoader(dataset=trainData, batch_size=int(self.batchSize), sampler=sampler)
        else:
            trainLoader = DataLoader(dataset=trainData, batch_size=int(self.batchSize), shuffle=True)
        # trainLoader = DataLoader(dataset=trainData, batch_size=1000, shuffle=True)
        stopTraining = False
        losses = []
        initLoss = 0
        for epoch in range(consts.maxEpochs):
            losses = []
            for inputBatch, targetBatch in trainLoader:
                inputBatch = inputBatch.to(self.device)
                targetBatch = targetBatch.to(self.device)
                predicted = torch.squeeze(self.model(inputBatch))
                loss = self.criterion(predicted, targetBatch)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (np.mean(losses) - self.bestLoss) < -self.minDelta:
                self.bestLoss = np.mean(losses)
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    stopTraining = True
                self.wait += 1
            if epoch == 0:
                initLoss = np.mean(losses)
            if stopTraining:
                print('    Stop criteria reached in epoch #{}'.format(epoch + 1))
                break
        print('    Initial loss: #{}, final loss: #{}'.format(initLoss, np.mean(losses)))

    def evalFunction(self, x):
        x = x.reshape(-1, self.inputSpaceDim)
        return self.model(Variable(torch.from_numpy(x.astype(np.float32)), requires_grad=False)).detach().numpy()

    def mutate(self):
        currentWeights = [layer.weight.data for layer in self.model.layers]
        currentHiddenSize = currentWeights[-2].shape[0]

        def addNodes():
            numNewNodes = np.random.choice([1, 2, 3, 4])
            newNodeHid2HidWeights = torch.zeros([numNewNodes, currentWeights[-2].shape[1]])
            nn.init.normal_(newNodeHid2HidWeights)
            newNodeOut2HidWeights = torch.zeros([currentWeights[-1].shape[0], numNewNodes])
            nn.init.normal_(newNodeOut2HidWeights)
            newParamsHid = torch.cat([currentWeights[-2], newNodeHid2HidWeights], dim=0)
            newParamsOut = torch.cat([currentWeights[-1], newNodeOut2HidWeights], dim=1)
            newLayerWidth = currentHiddenSize + numNewNodes
            return newParamsHid, newParamsOut, newLayerWidth

        def removeNode():
            idxsHidNodes = np.arange(currentWeights[-2].shape[0])
            nodeToDelete = np.random.choice(currentWeights[-2].shape[0], 1, replace=False)
            idxsHidNodes = np.delete(idxsHidNodes, nodeToDelete)
            newParamsHid = currentWeights[-2][idxsHidNodes, :]
            newParamsOut = currentWeights[-1][:, idxsHidNodes]
            newLayerWidth = currentHiddenSize - 1
            return newParamsHid, newParamsOut, newLayerWidth

        if random.random() >= 0.5:
            newWeightsHid, newWeightsOut, newLayerSize = addNodes()
        else:
            if currentHiddenSize > 1:
                newWeightsHid, newWeightsOut, newLayerSize = removeNode()
            else:
                newWeightsHid, newWeightsOut, newLayerSize = addNodes()
        self.model.layers[-2] = nn.Linear(currentWeights[-2].shape[1], newLayerSize)
        self.model.layers[-1] = nn.Linear(newLayerSize, currentWeights[-1].shape[0])
        self.model.layers[-2].weight.data = newWeightsHid.clone().detach().requires_grad_(True)
        self.model.layers[-1].weight.data = newWeightsOut.clone().detach().requires_grad_(True)
        self.specs['layerSizes'][-1] = newLayerSize


# noinspection PyUnresolvedReferences
class FIS(ANN):
    def __init__(self, inputSpaceDim):
        super().__init__(inputSpaceDim)
        self.name = 'FIS'

    def buildModel(self, inputSpaceDim, specs):
        return ANFIS(inputSpaceDim, specs)

    def getSpecs(self):
        specs = str(self.specs['numRules']) + ', ' + self.specs['membershipFunction'](1, 1).name + ', ' + str(
            self.specs['learnRate'])
        return specs

    def genRandomSpecs(self):
        numRules = np.random.choice(consts.feasibleNumRules)
        membershipFunction = np.random.choice(consts.feasibleMembershipFunctions)
        batchSize = np.random.choice(consts.feasibleBatchSizes)
        learnRate = np.random.choice(consts.feasibleLearnRates)
        specs = {'numRules': numRules,
                 'membershipFunction': membershipFunction,
                 'batchSize': batchSize,
                 'learnRate': learnRate}
        return specs

    def mutate(self):
        currentNumRules = self.model.membershipFunction.numRules

        def addRule():
            self.specs['numRules'] = self.specs['numRules'] + 1
            self.model.membershipFunction.numRules = currentNumRules + 1
            newRuleCenters = torch.randn(self.inputSpaceDim)
            newRuleWidths = torch.randn(self.inputSpaceDim)
            newRuleOutputWeights = torch.randn(1, 1)
            self.model.membershipFunction.centers = Parameter(
                torch.cat((self.model.membershipFunction.centers, newRuleCenters), dim=0), requires_grad=True)
            self.model.membershipFunction.widths = Parameter(
                torch.cat((self.model.membershipFunction.widths, newRuleWidths), dim=0), requires_grad=True)
            self.model.outputWeights = Parameter(
                torch.cat((self.model.outputWeights, newRuleOutputWeights), dim=1), requires_grad=True)

        # noinspection PyTypeChecker
        def removeRule():
            self.specs['numRules'] = self.specs['numRules'] - 1
            self.model.membershipFunction.numRules = currentNumRules - 1
            ruleToDelete = np.random.choice(currentNumRules)
            idxs = np.arange(currentNumRules * self.inputSpaceDim)
            idxsThatStay = np.delete(idxs, np.arange(ruleToDelete * self.inputSpaceDim,
                                                     (ruleToDelete + 1) * self.inputSpaceDim))
            self.model.membershipFunction.centers = Parameter(self.model.membershipFunction.centers[idxsThatStay],
                                                              requires_grad=True)
            self.model.membershipFunction.widths = Parameter(self.model.membershipFunction.widths[idxsThatStay],
                                                             requires_grad=True)
            self.model.outputWeights = Parameter(self.model.outputWeights[:, np.delete(np.arange(currentNumRules),
                                                                                       ruleToDelete)],
                                                 requires_grad=True)

        if random.random() >= 0.5:
            addRule()
        else:
            if currentNumRules > 1:
                removeRule()
            else:
                addRule()


class SVM(FunctionApproximator):
    def __init__(self, inputSpaceDim):
        super().__init__(inputSpaceDim)
        self.specs = self.genRandomSpecs()
        self.svr = None
        self.genModel()
        self.name = 'SVM'

    def genModel(self):
        if self.specs['kernel'] == 'poly':
            assert 'degree' in self.specs
            self.svr = SVR(kernel=str(self.specs['kernel']), degree=self.specs['degree'])
        else:
            self.svr = SVR(kernel=str(self.specs['kernel']))

    def getSpecs(self):
        specs = self.specs['kernel']
        if self.specs['kernel'] == 'poly':
            specs = specs + ', ' + str(self.specs['degree'])
        return specs

    def genRandomSpecs(self):
        specs = {}
        kernel = np.random.choice(consts.feasibleKernels)
        specs['kernel'] = kernel
        if kernel == 'poly':
            degree = np.random.choice(consts.feasibleDegrees)
            specs['degree'] = degree
        return specs

    def train(self, inputs, targets, weights=None):
        self.svr.fit(inputs, targets, sample_weight=weights)

    def evalFunction(self, x):
        x = x.reshape(-1, self.inputSpaceDim)
        y = self.svr.predict(x)
        y[y < 0] = 0
        return y

    def mutate(self):
        if self.specs['kernel'] == 'poly':
            if random.random() >= 0.5:
                self.specs['degree'] = max(self.specs['degree'] + np.random.choice([-1, 1]), 1)
            else:
                self.specs['kernel'] = np.random.choice(consts.feasibleKernels)
        else:
            # self.specs['kernel'] = np.random.choice(consts.feasibleKernels)
            # if self.specs['kernel'] == 'poly':
            #     degree = np.random.choice(consts.feasibleDegrees)
            #     self.specs['degree'] = degree
            pass
        self.genModel()


class KNNRegressor(FunctionApproximator):
    def __init__(self, inputSpaceDim):
        super().__init__(inputSpaceDim)
        self.specs = self.genRandomSpecs()
        self.knn = None
        self.genModel()
        self.name = 'KNNRegressor'

    def genModel(self):
        self.knn = KNeighborsRegressor(n_neighbors=self.specs['numNeighbors'], weights=self.specs['weights'])

    def getSpecs(self):
        specs = str(self.specs['numNeighbors']) + ', ' + self.specs['weights']
        return specs

    def genRandomSpecs(self):
        specs = {'numNeighbors': np.random.choice(consts.feasibleNumNeighbors),
                 'weights': np.random.choice(consts.feasibleWeights)}
        return specs

    def train(self, inputs, targets, weights=None):
        del weights
        self.knn.fit(inputs, targets)

    def evalFunction(self, x):
        x = x.reshape(-1, self.inputSpaceDim)
        y = self.knn.predict(x)
        y[y < 0] = 0
        return y

    def mutate(self):
        if random.random() >= 0.5:
            self.specs['numNeighbors'] = max(self.specs['numNeighbors'] + np.random.choice([-1, 1]), 1)
        else:
            self.specs['weights'] = np.random.choice(consts.feasibleWeights)
        self.genModel()


class DecTreeRegressor(FunctionApproximator):
    def __init__(self, inputSpaceDim):
        super().__init__(inputSpaceDim)
        self.specs = self.genRandomSpecs()
        self.tree = None
        self.genModel()
        self.name = 'DecTree'

    def genModel(self):
        self.tree = DecisionTreeRegressor(splitter=self.specs['splitter'], max_depth=self.specs['maxDepth'])

    def getSpecs(self):
        specs = str(self.specs['maxDepth']) + ', ' + self.specs['splitter']
        return specs

    def genRandomSpecs(self):
        # specs = {'maxDepth': np.random.choice(consts.feasibleMaxDepths),
        specs = {'maxDepth': None,
                 'splitter': np.random.choice(consts.feasibleSplitters)}
        return specs

    def train(self, inputs, targets, weights=None):
        self.tree.fit(inputs, targets, sample_weight=weights)

    def evalFunction(self, x):
        x = x.reshape(-1, self.inputSpaceDim)
        y = self.tree.predict(x)
        y[y < 0] = 0
        return y

    def mutate(self):
        self.genModel()


# noinspection PyUnresolvedReferences,PyTypeChecker
class MLP(nn.Module):
    def __init__(self, inputSpaceDim, outputSpaceDim, archSpecs):
        super(MLP, self).__init__()
        layerSizes = [inputSpaceDim] + archSpecs['layerSizes'] + [outputSpaceDim]
        useBias = archSpecs['useBias']
        self.activationFunctions = archSpecs['activationFunctions']
        assert len(self.activationFunctions) == (len(layerSizes) - 1)
        self.layers = nn.ModuleList()
        for l in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[l], layerSizes[l + 1],
                                         bias=useBias if l < (len(layerSizes) - 2) else True))
        self.layers[-1].bias.data.uniform_(0.1, 0.15)

    def forward(self, x):
        for l in range(len(self.layers)):
            activationFunction = self.activationFunctions[l]
            x = self.layers[l](x) if activationFunction == 'Linear' else activationFunction(self.layers[l](x))
        return x


class ANFIS(nn.Module):
    def __init__(self, inputSpaceDim, archSpecs):
        super().__init__()
        inputSpaceDim = inputSpaceDim
        numRules = archSpecs['numRules']
        membershipFunction = archSpecs['membershipFunction']

        self.membershipFunction = membershipFunction(numRules, inputSpaceDim)
        self.outputWeights = Parameter(torch.randn(1, numRules) + 1.0, requires_grad=True)

    def forward(self, x):
        rule = torch.prod(self.membershipFunction(x), dim=2)
        num = torch.sum(rule * self.outputWeights, dim=1)
        den = torch.clamp(torch.sum(rule, dim=1), 1e-12, 1e12)
        return torch.relu(num / den)
        # return num / den
