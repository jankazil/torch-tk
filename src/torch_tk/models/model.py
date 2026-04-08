'''
Abstract base class with instance methods required by torch-tk modules, in particular:

- constructor_dict()

which allows re-construting (cloning) a model instance in a given state.
'''

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class Model(torch.nn.Module, ABC):
    '''
    Abstract base class for models that can describe, save, restore, and reconstruct
    their instance in a given state.
    '''

    def __init__(self):
        '''
        Initialize the torch.nn.Module base class.
        '''
        super().__init__()

    @abstractmethod
    def forward(self, xb):
        '''
        Compute the forward pass for the given input batch.
        '''
        pass

    @abstractmethod
    def constructor_dict(self):
        '''
        Return a dictionary containing the positional and keyword arguments needed
        to reconstruct the model.
        '''
        pass

    def save_state_dict_to_file(self, path):
        '''
        Save the model state dictionary to a file, creating parent directories if needed.
        '''
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def save_to_file(self, path):
        '''
        Save to a file the constructor arguments and state dictionary needed to recreate an
        identical model instance.
        '''
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dictionary = {
            'constructor_dict': self.constructor_dict(),
            'state_dict': self.state_dict(),
        }

        torch.save(dictionary, path)

    @classmethod
    def load_from_file(cls, path, device=None):
        '''
        Create and initialize a model instance from a previously saved file.
        '''
        if not isinstance(path, Path):
            path = Path(path)

        dictionary = torch.load(path, map_location=device)

        model = cls.clone(
            dictionary['constructor_dict'],
            dictionary['state_dict'],
            device=device,
        )

        return model

    @classmethod
    def clone(cls, constructor_dict, state_dict, device=None):
        '''
        Reconstruct a model from its constructor arguments and state dictionary.
        '''
        args = constructor_dict.get('args', [])
        kwargs = constructor_dict.get('kwargs', {}).copy()

        if device is not None and 'device' in kwargs:
            kwargs['device'] = device

        model = cls(*args, **kwargs)
        model.load_state_dict(state_dict)

        if device is not None and 'device' not in kwargs:
            model = model.to(device)

        return model
