from modules.module import Module

"""
Implement **Rectified Linear Unit** non-linearity (aka **ReLU**): 
"""
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"
