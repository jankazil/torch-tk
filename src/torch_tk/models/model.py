'''
Abstract base class with instance methods required by torch-tk modules,
in particular:

- constructor_dict()
- state_dict()

which allows re-construting (cloning) a model instance in a given state.
'''

from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path

import torch


class Model(ABC):
    '''
    Abstract base class for models that can describe, save, restore, and reconstruct
    their instance in a given state.
    '''

    def __init__(self):
        '''
        Initialize the model in training mode.
        '''
        self.training = True

    @abstractmethod
    def forward(self, xb):
        '''
        Compute the forward pass for the given input batch.
        '''
        pass

    @abstractmethod
    def named_parameters(self):
        '''
        Return an iterator over tuples of the form (parameter name, parameter)
        for all trainable parameters in the model.
        '''
        pass

    @abstractmethod
    def constructor_dict(self):
        '''
        Return a dictionary containing the positional and keyword arguments needed
        to reconstruct the model.
        '''
        pass

    @abstractmethod
    def train(self, mode: bool = True):
        '''
        Set the model to training mode to the given training mode if it is given,
        and to True if the mode is not given.
        '''
        pass

    def eval(self):
        '''
        Set the model to evaluation mode.
        '''
        return self.train(False)

    def __call__(self, xb):
        return self.forward(xb)

    def parameters(self):
        '''
        Return all trainable parameters as a list.
        '''
        return [param for _, param in self.named_parameters()]

    def state_dict(self, keep_vars=False):
        '''
        Return a dictionary representing the model state.

        Currently this state consists of the named parameters. If additional
        objects become part of the model state, they should also be included
        here. The dictionary contains shallow copies of the parameters.
        '''
        if keep_vars:
            # Do not detach from autograd
            return dict(self.named_parameters())
        else:
            return {name: parameter.detach() for name, parameter in self.named_parameters()}

    def save_state_dict_to_file(self, path):
        '''
        Save the model state dictionary to a file, creating parent directories if needed.
        '''
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state_dict, strict=True):
        '''
        Copy parameter values from a state dictionary into this model instance.

        When 'strict' is True, missing or unexpected parameter names raise an
        error. Returns a named tuple containing missing and unexpected keys.
        '''

        expected_keys = {name for name, _ in self.named_parameters()}
        provided_keys = set(state_dict.keys())

        missing = list(expected_keys - provided_keys)
        unexpected = list(provided_keys - expected_keys)

        if strict:
            if missing:
                raise KeyError(f'Missing keys in state_dict: {missing}')
            if unexpected:
                raise KeyError(f'Unexpected keys in state_dict: {unexpected}')

        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name in state_dict:
                    if state_dict[name].shape != parameter.shape:
                        raise ValueError(
                            'Shape mismatch for ' + name + f': expected {parameter.shape}, got {state_dict[name].shape}'
                        )

                    parameter.copy_(state_dict[name].to(parameter.device, dtype=parameter.dtype))

        # Create named tuples of missing and unexpected keys

        IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])

        return IncompatibleKeys(missing_keys=missing, unexpected_keys=unexpected)

    def save_to_file(self, path):
        '''
        Save to a file the constructor arguments and state dictionary needed to recreate an
        identical model instance.
        '''

        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dictionary = {'constructor_dict': self.constructor_dict(), 'state_dict': self.state_dict()}

        torch.save(dictionary, path)

        return

    @classmethod
    def load_from_file(cls, path, device=None):
        '''
        Create and initialize a model instance from a previously saved file.
        '''

        if not isinstance(path, Path):
            path = Path(path)

        dictionary = torch.load(path, map_location=device)

        model = cls.clone(dictionary['constructor_dict'], dictionary['state_dict'], device=device)

        return model

    @classmethod
    def clone(cls, constructor_dict, state_dict, device=None):
        '''
        Reconstruct a model from its constructor arguments and state dictionary.
        '''

        args = constructor_dict['args']
        kwargs = constructor_dict['kwargs'].copy()

        if 'device' in kwargs:
            kwargs['device'] = device

        model = cls(*args, **kwargs)

        model.load_state_dict(state_dict)

        return model

    def to(self, device):
        '''
        Place the model named parameters to the given device.
        '''
        for name, parameter in self.named_parameters():
            setattr(self, name, parameter.to(device))

        self.device = device

        return self
