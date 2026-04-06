'''
Checkpoint utilities for saving and restoring self-describing model and optimizer state.

This module provides the CheckPointManager class, which saves checkpoints that
contain enough information to reconstruct both a model and its optimizer in the
state that created the checkpoint. Each checkpoint stores the fully qualified
class path, constructor arguments from constructor_dict(), and state from
state_dict() for both objects.

Models and optimizers must be importable from a stable class path and must expose

- constructor_dict()
- state_dict()
- load_state_dict()

A model must be reconstructible from its class path and constructor_dict().
An optimizer must be reconstructible from its class path, model.parameters(),
and constructor_dict().

The constructor data must be serializable. In practice, this means it should
contain only standard serializable Python values such as numbers, strings,
lists, tuples, and dictionaries.

This design is not suitable for optimizers that depend on non-serializable
constructor inputs, non-standard constructor signatures, custom parameter-group
reconstruction beyond model.parameters(), or runtime state not captured by
state_dict().
'''

from pathlib import Path

import torch

from .utils import class_path_of_instance, import_class


class CheckPointManager:
    '''
    Save checkpoints and rebuild a model and optimizer from a checkpoint file.

    The checkpoint contains the epoch, class paths, constructor arguments, and
    state dictionaries for the model and optimizer.
    '''

    def __init__(self, model, optimizer, directory):
        '''
        Store the model, optimizer, and checkpoint directory.

        The directory is converted to a Path and created if needed.
        '''
        if not isinstance(directory, Path):
            directory = Path(directory)

        self.model = model
        self.optimizer = optimizer
        self.directory = directory

    def save(self, epoch):
        '''
        Save a checkpoint for the current model and optimizer state.

        Returns the path to the written checkpoint file.
        '''
        if not hasattr(self.model, 'constructor_dict'):
            raise TypeError('Model must implement constructor_dict().')

        if not hasattr(self.optimizer, 'constructor_dict'):
            raise TypeError('Optimizer must implement constructor_dict().')

        checkpoint = {
            'epoch': epoch,
            'model_class_path': class_path_of_instance(self.model),
            'model_constructor_dict': self.model.constructor_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_class_path': class_path_of_instance(self.optimizer),
            'optimizer_constructor_dict': self.optimizer.constructor_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        file_name = Path(type(self.model).__name__ + '.' + type(self.optimizer).__name__ + '.epoch=' + str(epoch) + '.pt')

        self.directory.mkdir(parents=True, exist_ok=True)

        file_path = self.directory / file_name
        torch.save(checkpoint, file_path)

        return file_path

    @classmethod
    def load_from_file(cls, file_path, device=None):
        '''
        Load a checkpoint file and reconstruct the model, optimizer, and manager.

        Returns:
            checkpoint_manager, model, optimizer, epoch

        If device is given, the checkpoint is loaded onto that device and a
        saved model constructor argument named 'device' is overridden.
        '''
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        checkpoint = torch.load(file_path, map_location=device)

        # Reconstruct model

        model_class = import_class(checkpoint['model_class_path'])

        model_constructor_dict = checkpoint['model_constructor_dict']
        model_args = model_constructor_dict.get('args', [])
        model_kwargs = dict(model_constructor_dict.get('kwargs', {}))

        if 'device' in model_kwargs and device is not None:
            model_kwargs['device'] = device

        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model = model.to(device)

        # Reconstruct optimizer

        optimizer_class = import_class(checkpoint['optimizer_class_path'])

        optimizer_constructor_dict = checkpoint['optimizer_constructor_dict']
        optimizer_args = optimizer_constructor_dict.get('args', [])
        optimizer_kwargs = dict(optimizer_constructor_dict.get('kwargs', {}))

        optimizer = optimizer_class(model.parameters(), *optimizer_args, **optimizer_kwargs)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Create a checkpoint manager instance
        checkpoint_manager = cls(model, optimizer, file_path.parent)

        epoch = checkpoint['epoch']

        return checkpoint_manager, model, optimizer, epoch
