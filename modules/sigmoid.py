from modules.module import Module
import numpy as np
"""
Implement well-known **Sigmoid** non-linearity
"""

class Sigmoid(Module):
    def __init__(self):
         super(Sigmoid, self).__init__()
    
    def updateOutput(self, inpt):
        self.output = 1 / (1 + np.exp(-inpt))

        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * self.output * (1-self.output)
        return self.gradInput
    
    def __repr__(self):
        return "Sigmoid"