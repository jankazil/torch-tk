'''
Minimal manual stochastic gradient descent optimizer.

This module defines a lightweight optimizer class that applies
gradient descent updates directly to a supplied collection of
parameters. It also provides helpers for clearing gradients,
serializing constructor settings, and loading a minimal optimizer
state dictionary.
'''

import torch


class SGD_Manual:
    '''
    Minimal stochastic gradient descent optimizer that updates a
    supplied list of model parameters using a fixed learning rate.
    '''

    def __init__(self, parameters, lr=1e-1):
        '''
        Store the learning rate and store the incoming parameter
        iterable as a list for repeated optimizer operations.
        '''

        self.lr = lr  # Learning rate
        self.parameters = list(parameters)  # Model parameters

        return

    def zero_grad(self):
        '''
        Clear any existing gradients by setting each parameter's grad attribute to None.

        This can be more efficient than zeroing them with .zero_(). Also, it does not
        require first checking whether the gradients exist.
        '''

        for parameter in self.parameters:
            parameter.grad = None

        return

    def step(self):
        '''
        Update the model parameters using gradient descent with a given learning rate.
        '''

        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter -= parameter.grad * self.lr

        return

    def constructor_dict(self):
        '''
        Return a dictionary with the model constructor arguments and keyword arguments,
        which allow recreating this optimizer instance.
        '''
        return {
            'args': [],
            'kwargs': {
                'lr': self.lr,
            },
        }

    def state_dict(self):
        '''
        Return a minimal optimizer state dictionary whose structure
        mirrors torch.optim state_dict output, despite having no
        per-parameter optimizer state to store.
        '''
        param_ids = list(range(len(self.parameters)))
        return {
            'state': {},  # <- the per-parameter optimizer state would be stored here
            'param_groups': [
                {
                    'lr': self.lr,
                    'params': param_ids,
                }
            ],
        }

    def load_state_dict(self, state_dict):
        '''
        Chek for consistency a given optimizer state dictionary and load the
        stored learning rate into this instance.
        '''

        if 'param_groups' not in state_dict or 'state' not in state_dict:
            raise ValueError('Invalid optimizer state_dict')

        param_groups = state_dict['param_groups']
        if len(param_groups) != 1:
            raise ValueError('SGD_Manual expects exactly one parameter group')

        group = param_groups[0]
        if len(group['params']) != len(self.parameters):
            raise ValueError('Loaded state has different number of parameters')

        self.lr = group['lr']

        return
