'''
Abstract base class for models that describe their constructor arguments and
save and restore their state.

Model files contain the class path, constructor metadata, and state dictionary.
The concrete model class must remain importable, but callers do not need to know
it when loading through Model.load_from_file(). Older files are not supported.
'''

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch

from ..checkpoints.checkpoint_manager import _metadata_from_checkpoint, _metadata_to_checkpoint
from ..utilities.python import class_path_of_instance, import_class

_MODEL_FORMAT_VERSION = 2
_ModelType = TypeVar('_ModelType', bound='Model')


class Model(torch.nn.Module, ABC):
    '''
    Abstract base class for models that can describe, save, and reconstruct
    their constructor configuration and state dictionary.
    '''

    def __init__(self) -> None:
        '''
        Initialize the torch.nn.Module base class.
        '''
        # Initialize the base class.
        super().__init__()

    @abstractmethod
    def forward(self, xb: Any) -> Any:
        '''
        Compute the forward pass for the given input batch.

        Subclasses may specify more concrete input and output types.
        '''
        pass

    @abstractmethod
    def constructor_dict(self) -> dict[str, Any]:
        '''
        Return the positional and keyword arguments needed to reconstruct the model.

        Use 'args' for a list or tuple and 'kwargs' for a dictionary. Both entries
        are optional. File saving supports the constructor metadata types handled
        by checkpoint_manager, including numeric NumPy arrays, tensors, and paths.
        '''
        pass

    def save_state_dict_to_file(self, path: str | Path) -> None:
        '''
        Save only the state dictionary, creating parent directories if needed.

        This file does not contain the metadata required by load_from_file().
        '''
        # Create the parent directories.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the state dictionary.
        torch.save(self.state_dict(), path)

    def save_to_file(self, path: str | Path) -> None:
        '''
        Save the class path, constructor arguments, and state dictionary.

        Constructor metadata uses the same conversion as checkpoint_manager so
        that the file can be loaded with weights_only=True. The class definition
        and its dependencies must be available and importable when loading.

        The file contains no optimizer or training progress. Training/evaluation
        mode is selected explicitly when loading rather than saved in the file.
        Parent directories are created if needed.
        '''
        # Collect the model class, constructor arguments, and state dictionary.
        dictionary = {
            'format_version': _MODEL_FORMAT_VERSION,
            'model_class_path': class_path_of_instance(self),
            'model_constructor_dict': _metadata_to_checkpoint(self.constructor_dict()),
            'model_state_dict': self.state_dict(),
        }

        # Create the parent directories.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model information.
        torch.save(dictionary, path)

    @classmethod
    def load_from_file(
        cls: type[_ModelType],
        path: str | Path,
        *,
        training: bool,
        device: str | torch.device | None = None,
        allowed_model_class_paths: str | Iterable[str] | None = None,
    ) -> _ModelType:
        '''
        Reconstruct a model using the class recorded in a model file.

        Calling Model.load_from_file() does not require knowing the concrete
        model class. Calling a subclass method additionally requires the saved
        class to be that subclass or one of its subclasses.

        Args:
            path: File written by save_to_file().
            training: Required keyword-only Boolean. True selects training mode;
                False selects evaluation mode. The mode is applied recursively
                to all submodules. Evaluation mode does not disable gradients.
            device: Device for loading tensors and placing the model. If given,
                also overrides an explicit constructor parameter named 'device'.
                If None, saved tensor locations and constructor devices are used.
            allowed_model_class_paths: Optional class path or collection of class
                paths permitted for reconstruction. Checked before importing.

        Returns:
            The reconstructed model with the requested training/evaluation mode.

        Only load trusted files: weights_only=True restricts deserialization,
        but class reconstruction still imports and runs Python code.

        This method rejects older model files.

        This method accepts checkpoint files created by the torch_tk
        CheckPointManager, restoring only the model.
        '''
        # Check the requested training/evaluation mode before loading.
        if not isinstance(training, bool):
            raise TypeError('training must be a bool.')

        # Load the model information.
        dictionary = torch.load(Path(path), map_location=device, weights_only=True)

        # Check the model file format.
        if not isinstance(dictionary, dict):
            raise TypeError('Model file must contain a dictionary.')

        version = dictionary.get('format_version')
        if type(version) is not int or version != _MODEL_FORMAT_VERSION:
            raise ValueError(f'Unsupported model format version. Expected {_MODEL_FORMAT_VERSION}, got {version!r}.')

        required_keys = {
            'model_class_path',
            'model_constructor_dict',
            'model_state_dict',
        }
        if not required_keys.issubset(dictionary):
            raise ValueError('Model file is missing required model information.')

        # Check the saved class path before importing.
        model_class_path = dictionary['model_class_path']
        if not isinstance(model_class_path, str) or not model_class_path:
            raise TypeError('Model class path must be a nonempty string.')

        if allowed_model_class_paths is not None:
            if isinstance(allowed_model_class_paths, str):
                allowed_model_class_paths = {allowed_model_class_paths}
            else:
                allowed_model_class_paths = set(allowed_model_class_paths)

            if model_class_path not in allowed_model_class_paths:
                raise ValueError(f'Model class path is not allowed: {model_class_path!r}')

        # Import the model class and check compatibility with the calling class.
        model_class = import_class(model_class_path)
        if not inspect.isclass(model_class) or not issubclass(model_class, cls):
            raise TypeError(
                f'Saved model class {model_class_path!r} must be a subclass of {cls.__module__}.{cls.__qualname__}.'
            )

        # Restore the constructor metadata and reconstruct the model.
        constructor_dict = _metadata_from_checkpoint(dictionary['model_constructor_dict'])
        model = model_class.clone(
            constructor_dict,
            dictionary['model_state_dict'],
            device=device,
        )

        # Apply the requested mode to the model and its submodules.
        model.train(training)

        return model

    @classmethod
    def clone(
        cls: type[_ModelType],
        constructor_dict: Mapping[str, Any],
        state_dict: Mapping[str, Any],
        device: str | torch.device | None = None,
    ) -> _ModelType:
        '''
        Reconstruct a model from its constructor arguments and state dictionary.

        If device is given, override an explicit constructor parameter named
        'device', whether supplied positionally or by keyword, and move the
        reconstructed model to that device. A device entry passed through
        **kwargs is also overridden if present in the saved arguments.

        The model retains the mode set by its constructor. load_from_file()
        subsequently applies the mode requested by its required training argument.
        '''
        # Check and copy the constructor arguments.
        if not isinstance(constructor_dict, Mapping):
            raise TypeError('Model constructor information must be a mapping.')

        args = list(constructor_dict.get('args', []))
        kwargs = dict(constructor_dict.get('kwargs', {}))

        # Override the constructor device without duplicating positional arguments.
        if device is not None:
            signature = inspect.signature(cls)
            parameter = signature.parameters.get('device')

            if parameter is not None and parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                bound.arguments['device'] = device
                args = list(bound.args)
                kwargs = dict(bound.kwargs)
            elif 'device' in kwargs:
                kwargs['device'] = device

        # Construct the model and restore its state dictionary.
        model = cls(*args, **kwargs)
        model.load_state_dict(state_dict)

        # Move the model to the requested device.
        if device is not None:
            model = model.to(device)

        return model
