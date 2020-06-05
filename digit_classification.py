"""
We are using well known *MNIST*(http://yann.lecun.com/exdb/mnist/) as our dataset.
Lets start with *cool visualization*(http://scs.ryerson.ca/~aharley/vis/). 
The most beautiful demo is the second one, if you are not familiar with 
convolutions you can return to it in further lectures lectures. 
"""

import os
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Fetch MNIST dataset and create a local copy.
if os.path.exists('mnist.npz'):
    # data = np.load('mnist.npz',)
    with np.load('mnist.npz', 'r',allow_pickle=True) as data:
        X = data['X']
        y = data['y']
else:
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # X, y = mnist.data / 255.0, mnist.target
    np.savez('mnist.npz', X=X, y=y)

print("data shape:", X.shape, y.shape)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
YOUR TASKS:
- **Compare** `ReLU`,`Sigmoid`, `SoftPlus` activation functions. 
You would better pick the best optimizer params for each of them, but it is 
overkill for now. Use an architecture of your choice for the comparison and let
it be fixed.

- **Try** inserting `BatchMeanSubtraction` between `Dense` module and 
  activation functions.

- Plot the losses both from activation functions comparison and 
  `BatchMeanSubtraction` comparison on one plot. Please find a scale (log?) 
  when the lines are distinguishable, do not forget about naming the axes, 
  the plot should be goodlooking. You can submit pictures of this plots.

- Write your personal opinion on the activation functions, think about 
  computation times too. Does `BatchMeanSubtraction` help?

- **Finally**, use all your knowledge to build a super cool model on this 
  dataset, do not forget to split dataset into train and validation. Use 
  **dropout** to prevent overfitting, play with **learning rate decay**. 
  You can use **data augmentation** such as rotations, translations to boost 
  your score. Use your knowledge and imagination to train a model. 

- Print your accuracy at the end of the code. Also write down the best accuracy 
  that you could get on test setIt should be around 90%.

- Hint: logloss for MNIST should be around 0.5.

- Suggestions: it can be easyer to use jupyter notebook for experimenting,
  but final results MUST be in this file (or multiple files)

Write down all your answers at the end of the file as a comment.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Your code goes here. 

# ...

# Your answers here