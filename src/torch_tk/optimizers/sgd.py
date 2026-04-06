'''
Custom SGD optimizer wrapper built on torch.optim.SGD.

This module defines an SGD optimizer subclass that stores its
constructor arguments on the instance and provides a
constructor_dict method for reconstructing the optimizer
configuration without the params argument.
'''

import torch


class SGD(torch.optim.SGD):
    '''
    SGD optimizer subclass that preserves its constructor settings
    and can export them in a reconstruction-friendly form.
    '''

    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
    ):
        '''
        Initialize the optimizer and store all constructor arguments
        needed to recreate its configuration later.
        '''

        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.foreach = foreach
        self.differentiable = differentiable
        self.fused = fused

    def constructor_dict(self):
        '''
        Return the stored optimizer constructor arguments in a
        dictionary, excluding the params argument.
        '''
        return {
            'args': [],
            'kwargs': {
                'lr': self.lr,
                'momentum': self.momentum,
                'dampening': self.dampening,
                'weight_decay': self.weight_decay,
                'nesterov': self.nesterov,
                'maximize': self.maximize,
                'foreach': self.foreach,
                'differentiable': self.differentiable,
                'fused': self.fused,
            },
        }
