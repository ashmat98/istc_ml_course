import unittest
import numpy as np

from modules import *

def check_grad(mdl: Module, *args, input_shape=(2,3),h = 1e-7):
    """
    This function is for testing. Do not change this.
    """
    inpt = np.random.randn(*input_shape)

    output = mdl.forward(inpt,*args)
    output_shape = output.shape

    gradOutput = np.random.randn(*output_shape)

    gradInput = mdl.backward(inpt, gradOutput)

    gradInputEstimate = np.zeros_like(gradInput)

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            total_grad_estimate = 0
            for k in range(output_shape[0]):
                for l in range(output_shape[1]):

                    inpt_d = inpt.copy()
                    inpt_d[i,j] += h
                    output_d = mdl.forward(inpt_d)
                    grad_estimate = ((output_d - output)/h)[k,l]
                    total_grad_estimate += grad_estimate * gradOutput[k,l]
            gradInputEstimate[i,j] = total_grad_estimate
    return gradInput, gradInputEstimate



EPS = 1e-7

input_shape = (5,6)

class TestModules(unittest.TestCase):
    def test_sigmoid(self):
        mdl = Sigmoid()

        output = mdl.forward(np.arange(-6,6).reshape(3,4))
        self.assertTrue(np.isclose(output,
        [[0.00247262, 0.00669285, 0.01798621, 0.04742587],
         [0.11920292, 0.26894142, 0.5       , 0.73105858],
         [0.88079708, 0.95257413, 0.98201379, 0.99330715]]).all())

        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)


        


    def test_dense(self):
        mdl = Dense(input_shape[1], input_shape[1]+1)
        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)

    def test_relu(self):
        mdl = ReLU()

        output = mdl.forward(np.arange(-6,6).reshape(3,4))
        self.assertTrue(np.isclose(output,
        [[-0., -0., -0., -0.],
         [-0., -0.,  0.,  1.],
         [ 2.,  3.,  4.,  5.]]).all())

        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)

        


    def test_softmax(self):
        mdl = SoftMax()

        output = mdl.forward(np.arange(-6,6).reshape(3,4))
        self.assertTrue(np.isclose(output,
        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]).all())

        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)

        

    def test_softplus(self):
        mdl = SoftPlus()

        output = mdl.forward(np.arange(-6,6).reshape(3,4))
        self.assertTrue(np.isclose(output,
        [[2.47568514e-03, 6.71534849e-03, 1.81499279e-02, 4.85873516e-02],
         [1.26928011e-01, 3.13261688e-01, 6.93147181e-01, 1.31326169e+00],
         [2.12692801e+00, 3.04858735e+00, 4.01814993e+00, 5.00671535e+00]]).all())

        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)


    def test_leakyrelu(self):
        mdl = LeakyReLU(slope=0.4)

        output = mdl.forward(np.arange(-6,6).reshape(3,4))
        self.assertTrue(np.isclose(output,
        [[-2.4, -2. , -1.6, -1.2],
         [-0.8, -0.4,  0. ,  1. ],
         [ 2. ,  3. ,  4. ,  5. ]]).all())

        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)


    def test_tanh(self):
        mdl = Tanh()

        output = mdl.forward(np.arange(-6,6).reshape(3,4))
        self.assertTrue(np.isclose(output,
        [[-0.99998771, -0.9999092 , -0.9993293 , -0.99505475],
         [-0.96402758, -0.76159416,  0.        ,  0.76159416],
         [ 0.96402758,  0.99505475,  0.9993293 ,  0.9999092 ]]).all())

        grad, grad_estimate = check_grad(mdl, input_shape=input_shape)
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)

