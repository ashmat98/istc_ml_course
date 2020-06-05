from modules.module import Module

"""
Implement **Leaky Rectified Linear Unit**
(http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). 
Expriment with slope. 
"""

class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"
