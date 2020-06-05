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
    
    @staticmethod
    def softmax(I):
        E = np.exp(I)
        return E / np.sum(E, axis=1, keepdims=True)

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        self.output = SoftMax.softmax(inpt)
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        sigma = self.output
        self.gradInput = sigma * (gradOutput - np.sum(gradOutput*sigma, 
                                                      axis=1, keepdims=True))
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"