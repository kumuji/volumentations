"""Fast augmentations for 3d data."""

from importlib.metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from .augmentations.transforms import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *
