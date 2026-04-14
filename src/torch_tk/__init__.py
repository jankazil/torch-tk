from importlib.metadata import PackageNotFoundError, version

_DIST_NAME = "torch-tk"

try:
    __version__ = version(_DIST_NAME)
except PackageNotFoundError:
    pkg = __package__ or __name__.split(".", 1)[0]
    try:
        __version__ = version(pkg)
    except PackageNotFoundError:
        __version__ = "0.0.0+local"

from .models.model import Model
from .optimizers.sgd import SGD
from .optimizers.adam import Adam
from .training.trainer import Trainer
from .checkpoints.checkpoint_manager import CheckPointManager
from .diagnostics.diagnostics import Diagnostics

__all__ = [
    "__version__",
    "Model",
    "SGD",
    "Adam",
    "Trainer",
    "CheckPointManager",
    "Diagnostics",
]
