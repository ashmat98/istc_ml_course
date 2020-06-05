from modules.module import Module
import numpy as np
"""
This one is probably the hardest but as others only takes 5 lines of code in total. 
- input:   **batch_size x n_feats**
- output: **batch_size x n_feats**
"""
class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"
