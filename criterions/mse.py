from criterions.criterion import Criterion
import numpy as np

"""
The **MSECriterion**, which is basic L2 norm usually used for regression.
"""
class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, inpt, target):
        # <Your Code Goes Here>
        raise NotImplementedError()

        return self.output 
 
    def updateGradInput(self, inpt, target):
        # <Your Code Goes Here>
        raise NotImplementedError()

        return self.gradInput

    def __repr__(self):
        return "MSECriterion"
