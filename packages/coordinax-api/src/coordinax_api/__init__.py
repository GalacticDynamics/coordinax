"""Abstract dispatch API for `coordinax`.

This package defines the abstract dispatch interfaces for `coordinax`'s core
functionality.
"""

__all__ = (
    "__version__",
    "vconvert",
)

from ._vector import vconvert
from ._version import version as __version__
