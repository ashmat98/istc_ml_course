import numpy as np
"""
In order to implement this, look at the **Algorithm 1** on page 2 of this paper:
(https://arxiv.org/pdf/1412.6980.pdf).
"""

def adam_optimizer(x, dx, config, state):
    state.setdefault('old_grad', {})
    state.setdefault('old_grad_square', {})
    i = 0
    for cur_layer_x, cur_layer_dx in zip(x,dx):
        for cur_x, cur_dx in zip(cur_layer_x,cur_layer_dx):
            
            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            cur_old_grad_square = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            
            cur_old_grad = config['beta1'] * cur_old_grad + (1 - config['beta1']) * cur_dx
            cur_old_grad_square = config['beta2'] * cur_old_grad_square + (1 - config['beta2']) * cur_dx * cur_dx
            m_hat = cur_old_grad/(1-config['beta1']**(i+1))
            v_hat = cur_old_grad_square/(1-config['beta2']**(i+1))
            if cur_old_grad.shape[0] == 1:
                cur_x = cur_x.reshape(cur_old_grad.shape)
            
            np.add(cur_x, -config['learning_rate'] * m_hat / (v_hat**0.5 + config['epsilon']), out=cur_x)
            i += 1