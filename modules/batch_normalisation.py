from modules.module import Module
import numpy as np

"""
One of the most significant recent ideas that impacted NNs a lot is 
**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple,
yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the 
way through NN. This improves the convergence for deep models letting it train
them for days but not weeks. **You are** to implement a part of the layer: mean
subtraction. That is, the module should calculate mean value for every feature
(every column) and subtract it.

Note, that you need to estimate the mean over the dataset to be able to predict
on test examples. The right way is to create a variable which will hold smoothed
mean over batches (exponential smoothing works good) and use it when forwarding
test examples.

When training calculate mean as folowing: 
```
    mean_to_subtract = self.old_mean * alpha + batch_mean * (1 - alpha)
```
and do backpropagation accordingly.

when evaluating (`self.training == False`) set $alpha = 1$.

- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**
"""

class BatchMeanSubtraction(Module):
    def __init__(self, alpha = 0.95):
        super(BatchMeanSubtraction, self).__init__()
        
        self.alpha = alpha
        self.old_mean = None 
        
    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        raise NotImplementedError()
        return self.gradInput
    
    def __repr__(self):
        return "BatchMeanNormalization"
