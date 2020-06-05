from modules.module import Module
import numpy as np
"""
Implement **SoftPlus**
(https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations.
Look, how they look a lot like ReLU.
"""

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, inpt):
        self.output = np.log(1 + np.exp(inpt))

        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput / (1 + np.exp(-inpt))

        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"