"""
This part is **OPTIONAL**, you may not do it, but it is easy and extremely
interesting.
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Now we are going to build a cool model, named AUTOENCODER. 
The aim is simple: **encode** the data to a lower dimentional representation. 
Why? Well, if we can **decode** this representation back to original data with 
"small" reconstuction loss then we can store only compressed representation 
saving memory. But the most important thing is -- we can reuse trained 
autoencoder for classification. 

see picture https://multithreaded.stitchfix.com/assets/images/blog/PS_NN_graphic_colors2.png
from this blogpost https://multithreaded.stitchfix.com/blog/2015/09/17/deep-style/
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""
Now implement an autoencoder:

Build it such that dimetionality inside autoencoder changes like that: 

784 (data) -> 512 -> 256 -> 128 -> 30 -> 128 -> 256 -> 512 -> 784

Use **MSECriterion** to score the reconstruction. Use **BatchMeanNormalization**
between **Dense** and **ReLU**. You may not use nonlinearity in bottleneck layer.

You may train it for 9 epochs with batch size = 256, initial lr = 0.1 droping by
a factor of 2 every 3 epochs. The reconstruction loss should be about 6.0 and
visual quality decent already.
Do not spend time on changing architecture, they are more or less the same.
"""

# Your code goes here. 

"""
Some time ago NNs were a lot poorer and people were struggling to learn deep 
models. To train a classification net people were training autoencoder first 
(to train autoencoder people were pretraining single layers with RBM
(https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)), then substituting
the decoder part with classification layer (yeah, they were struggling with 
training autoencoders a lot, and complex techniques were used at that dark 
times). We are going to this now, fast and easy. 
"""

# Extract inner representation for train and validation, 
# you should get (n_samples, 30) matrices
# <Your code goes here>


# Now build a logistic regression or small classification net
# Learn the weights
# <Your code goes here>

# Now chop off decoder part
# (you may need to implement `remove` method for Sequential container) 
# <Your code goes here>

# And add learned layers ontop.

# Now optimize whole model
# <Your code goes here>

"""
What do you think, does it make sense to build real-world classifiers this way? 
Did it work better for you than a straightforward one? Looks like it was not 
the same ~8 years ago, what has changed beside computational power?

<your answer here>
"""

"""
Run PCA with 30 components on the *train set*, plot original image, autoencoder
and PCA reconstructions side by side for 10 samples from *validation set*.
Probably you need to use the following snippet to make aoutpencoder examples look comparible.
"""
# np.clip(prediction,0,1)
# <Your code goes here>