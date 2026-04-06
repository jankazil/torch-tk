'''
Custom Adam optimizer wrapper built on torch.optim.Adam.

This module defines an Adam optimizer subclass that stores its
constructor arguments on the instance and provides a
constructor_dict method for reconstructing the optimizer
configuration without the params argument.
'''

import torch


class Adam(torch.optim.Adam):
    '''
    Adam optimizer subclass that preserves its constructor settings
    and can export them in a reconstruction-friendly form.
    '''

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        decoupled_weight_decay=False,
    ):
        '''
        Initialize the optimizer and store all constructor arguments
        needed to recreate its configuration later.
        '''

        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=decoupled_weight_decay,
        )

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
        self.fused = fused
        self.decoupled_weight_decay = decoupled_weight_decay

    def constructor_dict(self):
        '''
        Return the stored optimizer constructor arguments in a
        dictionary, excluding the params argument.
        '''
        return {
            'args': [],
            'kwargs': {
                'lr': self.lr,
                'betas': self.betas,
                'eps': self.eps,
                'weight_decay': self.weight_decay,
                'amsgrad': self.amsgrad,
                'foreach': self.foreach,
                'maximize': self.maximize,
                'capturable': self.capturable,
                'differentiable': self.differentiable,
                'fused': self.fused,
                'decoupled_weight_decay': self.decoupled_weight_decay,
            },
        }
