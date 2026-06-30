'''
Checkpoint utilities for saving and restoring self-describing model and optimizer state.

This module provides the CheckPointManager class, which saves checkpoints that
contain enough information to reconstruct both a model and its optimizer in the
state that created the checkpoint.

The checkpoint format intentionally stores constructor metadata in a form that
can be loaded with torch.load(..., weights_only=True). NumPy arrays and NumPy
scalars in constructor metadata are converted at save time and restored before
calling model and optimizer constructors.

This version is not compatible with checkpoints saved by older versions of this
module.
'''

import inspect
from pathlib import Path

import torch

from .utils import class_path_of_instance, import_class

_CHECKPOINT_FORMAT_VERSION = 2
_METADATA_TYPE_KEY = '__torch_tk_checkpoint_metadata_type__'


def _numpy_or_none():
    '''
    Return the NumPy module if available, otherwise None.

    NumPy is optional for checkpoints that do not contain NumPy metadata.
    '''
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    return np


def _require_numpy():
    '''
    Return the NumPy module, or raise a clear error if it is unavailable.
    '''
    np = _numpy_or_none()

    if np is None:
        raise RuntimeError('This checkpoint contains NumPy metadata, but NumPy is not installed.')

    return np


def _metadata_to_checkpoint(value):
    '''
    Convert constructor metadata to values compatible with
    torch.load(..., weights_only=True).

    Supported metadata types are:
    - None
    - bool, int, float, str
    - torch.Tensor
    - torch.device
    - pathlib.Path
    - NumPy numeric or boolean arrays
    - NumPy scalar values
    - lists
    - tuples
    - dictionaries with simple keys
    '''
    np = _numpy_or_none()

    if np is not None:
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'buiuf':
                raise TypeError(
                    'Only boolean, integer, unsigned integer, and floating-point '
                    'NumPy arrays are supported in checkpoint constructor '
                    f'metadata. Got dtype {value.dtype!r}.'
                )

            return {
                _METADATA_TYPE_KEY: 'numpy.ndarray',
                'dtype': value.dtype.str,
                'shape': list(value.shape),
                'data': value.reshape(-1).tolist(),
            }

        if isinstance(value, np.generic):
            return _metadata_to_checkpoint(value.item())

    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, torch.device):
        return {
            _METADATA_TYPE_KEY: 'torch.device',
            'value': str(value),
        }

    if isinstance(value, Path):
        return {
            _METADATA_TYPE_KEY: 'pathlib.Path',
            'value': str(value),
        }

    if isinstance(value, tuple):
        return {
            _METADATA_TYPE_KEY: 'tuple',
            'items': [_metadata_to_checkpoint(item) for item in value],
        }

    if isinstance(value, list):
        return [_metadata_to_checkpoint(item) for item in value]

    if isinstance(value, dict):
        converted = {}

        for key, item in value.items():
            if not isinstance(key, str | int | float | bool | None):
                raise TypeError(
                    'Checkpoint constructor metadata dictionaries may only use '
                    'None, bool, int, float, and str keys. Got key type '
                    f'{type(key).__module__}.{type(key).__qualname__}.'
                )

            converted[key] = _metadata_to_checkpoint(item)

        return converted

    raise TypeError(f'Unsupported checkpoint constructor metadata type: {type(value).__module__}.{type(value).__qualname__}')


def _metadata_from_checkpoint(value):
    '''
    Restore constructor metadata converted by _metadata_to_checkpoint().
    '''
    if isinstance(value, list):
        return [_metadata_from_checkpoint(item) for item in value]

    if isinstance(value, dict):
        metadata_type = value.get(_METADATA_TYPE_KEY)

        if metadata_type is None:
            return {key: _metadata_from_checkpoint(item) for key, item in value.items()}

        if metadata_type == 'numpy.ndarray':
            np = _require_numpy()

            return np.asarray(
                value['data'],
                dtype=np.dtype(value['dtype']),
            ).reshape(tuple(value['shape']))

        if metadata_type == 'torch.device':
            return torch.device(value['value'])

        if metadata_type == 'pathlib.Path':
            return Path(value['value'])

        if metadata_type == 'tuple':
            return tuple(_metadata_from_checkpoint(item) for item in value['items'])

        raise ValueError(f'Unsupported checkpoint metadata tag: {metadata_type!r}')

    return value


def _check_checkpoint_format(checkpoint):
    '''
    Reject checkpoints that do not use this module's current format.
    '''
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must contain a dictionary.')

    version = checkpoint.get('format_version')

    if version != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            'Unsupported checkpoint format version. Expected '
            f'{_CHECKPOINT_FORMAT_VERSION}, got {version!r}. This loader is '
            'not compatible with checkpoints saved by older versions.'
        )


def _check_allowed_class_path(class_path, allowed_class_paths, kind):
    '''
    Optionally reject checkpoint class paths that are not explicitly allowed.
    '''
    if allowed_class_paths is None:
        return

    if isinstance(allowed_class_paths, str):
        allowed_class_paths = {allowed_class_paths}
    else:
        allowed_class_paths = set(allowed_class_paths)

    if class_path not in allowed_class_paths:
        raise ValueError(f'Checkpoint {kind} class path is not allowed: {class_path!r}')


class CheckPointManager:
    '''
    Save checkpoints and rebuild a model and optimizer from a checkpoint file.

    The checkpoint contains the epoch, batch size, class paths, constructor
    arguments, and state dictionaries for the model and optimizer.
    '''

    def __init__(self, model, optimizer, directory):
        '''
        Store the model, optimizer, and checkpoint directory.

        The directory is converted to a Path.
        '''
        if not isinstance(directory, Path):
            directory = Path(directory)

        self.model = model
        self.optimizer = optimizer
        self.directory = directory

        self.suffix = '.pt'

    def save(self, epoch, batch_size):
        '''
        Save a checkpoint for the current model and optimizer state.

        Returns the path to the written checkpoint file.
        '''
        if not hasattr(self.model, 'constructor_dict'):
            raise TypeError('Model must implement constructor_dict().')

        if not hasattr(self.optimizer, 'constructor_dict'):
            raise TypeError('Optimizer must implement constructor_dict().')

        checkpoint = {
            'format_version': _CHECKPOINT_FORMAT_VERSION,
            'epoch': _metadata_to_checkpoint(epoch),
            'batch_size': _metadata_to_checkpoint(batch_size),
            'model_class_path': class_path_of_instance(self.model),
            'model_constructor_dict': _metadata_to_checkpoint(self.model.constructor_dict()),
            'model_state_dict': self.model.state_dict(),
            'optimizer_class_path': class_path_of_instance(self.optimizer),
            'optimizer_constructor_dict': _metadata_to_checkpoint(self.optimizer.constructor_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        file_name = 'checkpoint.epoch=' + str(epoch) + self.suffix

        self.directory.mkdir(parents=True, exist_ok=True)

        file_path = self.directory / file_name
        torch.save(checkpoint, file_path)

        return file_path

    def list_checkpoint_files(self):
        '''
        Return checkpoint file paths in the checkpoint directory.
        '''
        return sorted(self.directory.glob('*' + self.suffix))

    @classmethod
    def load_from_file(
        cls,
        file_path,
        device=None,
        allowed_model_class_paths=None,
        allowed_optimizer_class_paths=None,
    ):
        '''
        Load a checkpoint file and reconstruct the model, optimizer, and manager.

        Returns:
            checkpoint_manager, model, optimizer, epoch, batch_size

        If device is given, the checkpoint is loaded onto that device and a
        saved model constructor argument named 'device' is overridden.

        The optional allowed_model_class_paths and allowed_optimizer_class_paths
        arguments can be used to restrict which classes may be imported and
        instantiated from checkpoint metadata.
        '''
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        checkpoint = torch.load(
            file_path,
            map_location=device,
            weights_only=True,
        )

        _check_checkpoint_format(checkpoint)

        model_class_path = checkpoint['model_class_path']
        _check_allowed_class_path(
            model_class_path,
            allowed_model_class_paths,
            'model',
        )

        model_class = import_class(model_class_path)

        model_constructor_dict = _metadata_from_checkpoint(checkpoint['model_constructor_dict'])
        model_args = model_constructor_dict.get('args', [])
        model_kwargs = dict(model_constructor_dict.get('kwargs', {}))

        model_signature = inspect.signature(model_class)

        if device is not None and 'device' in model_signature.parameters:
            model_kwargs['device'] = device

        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model = model.to(device)

        optimizer_class_path = checkpoint['optimizer_class_path']
        _check_allowed_class_path(
            optimizer_class_path,
            allowed_optimizer_class_paths,
            'optimizer',
        )

        optimizer_class = import_class(optimizer_class_path)

        optimizer_constructor_dict = _metadata_from_checkpoint(checkpoint['optimizer_constructor_dict'])
        optimizer_args = optimizer_constructor_dict.get('args', [])
        optimizer_kwargs = dict(optimizer_constructor_dict.get('kwargs', {}))

        optimizer = optimizer_class(
            model.parameters(),
            *optimizer_args,
            **optimizer_kwargs,
        )

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        checkpoint_manager = cls(model, optimizer, file_path.parent)

        epoch = _metadata_from_checkpoint(checkpoint['epoch'])
        batch_size = _metadata_from_checkpoint(checkpoint['batch_size'])

        return checkpoint_manager, model, optimizer, epoch, batch_size
