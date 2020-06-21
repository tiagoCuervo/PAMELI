from optproblems import dtlz, wfg
from platypus import Problem, Real


class myDTLZ1:
    def __init__(self, nvars=None, nobjs=2):
        k = 5
        if nvars is None:
            self.nvars = nobjs + k - 1
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ1(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ1'


class myDTLZ2:
    def __init__(self, nvars=None, nobjs=2):
        if nvars is None:
            self.nvars = nobjs + 9
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ2(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ2'


class myDTLZ3:
    def __init__(self, nvars=None, nobjs=2):
        if nvars is None:
            self.nvars = nobjs + 9
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ3(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ3'


class myDTLZ4:
    def __init__(self, nvars=None, nobjs=2):
        if nvars is None:
            self.nvars = nobjs + 9
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ4(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ4'


class myDTLZ5:
    def __init__(self, nvars=None, nobjs=2):
        if nvars is None:
            self.nvars = nobjs + 9
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ5(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ5'


class myDTLZ6:
    def __init__(self, nvars=None, nobjs=2):
        if nvars is None:
            self.nvars = nobjs + 9
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ6(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ6'


class myDTLZ7:
    def __init__(self, nvars=None, nobjs=2):
        if nvars is None:
            self.nvars = nobjs + 19
        else:
            self.nvars = nvars
        self.nobjs = nobjs
        self.F = dtlz.DTLZ7(self.nobjs, self.nvars)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'DTLZ7'


class myWFG1:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG1(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG2:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG2(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG3:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG3(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG4:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG4(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG5:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG5(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG6:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG6(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG7:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG7(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG8:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG8(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

class myWFG9:
    def __init__(self, nvars=None, nobjs=2):
        self.nvars = 10
        self.nobjs = 2
        self.F = wfg.WFG9(num_objectives=2, num_variables=10, k=4)
        self.problem = Problem(self.nvars, self.nobjs)
        self.problem.types[:] = Real(0, 1)
        self.problem.function = self.F
        self.problem.name = 'WFG1'

