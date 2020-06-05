from modules.module import Module
import numpy as np
"""
Implement well-known **Sigmoid** non-linearity
"""

class Sigmoid(Module):
    def __init__(self):
         super(Sigmoid, self).__init__()
    
    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        raise NotImplementedError() 
        return self.gradInput
    
    def __repr__(self):
        return "Sigmoid"
