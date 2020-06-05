from modules.module import Module
import numpy as np 
"""
Implement **hyperbolic tangent** non-linearity (aka **Tanh**): 
Note that Tanh is scaled version of the sigmoid function.
"""
class Tanh(Module):
    def __init__(self):
         super(Tanh, self).__init__()
    
    def updateOutput(self, inpt):
        self.output = np.tanh(inpt)  
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * (1-np.square(self.output))
        return self.gradInput
    
    def __repr__(self):
        return "Tanh"