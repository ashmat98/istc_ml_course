from modules.module import Module

"""
Implement **Rectified Linear Unit** non-linearity (aka **ReLU**): 
"""
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, inpt):
        self.output = inpt * (inpt>0).astype(float)

        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * (inpt>0).astype(float)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"