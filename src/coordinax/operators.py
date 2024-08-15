"""Transformations of the coordinate reference frame.

E.g. a translation.
"""
# pylint: disable=unused-wildcard-import,wildcard-import

from ._coordinax.operators import (
    _base,
    _composite,
    _galilean,
    _identity,
    _sequential,
    funcs,
)
from ._coordinax.operators._composite import *  # noqa: F403
from ._coordinax.operators._galilean import *  # noqa: F403
from ._coordinax.operators._identity import *  # noqa: F403
from ._coordinax.operators._sequential import *  # noqa: F403
from ._coordinax.operators.base import *  # noqa: F403
from ._coordinax.operators.funcs import *  # noqa: F403

__all__: list[str] = []
__all__ += _base.__all__
__all__ += _composite.__all__
__all__ += _sequential.__all__
__all__ += _identity.__all__
__all__ += _galilean.__all__
__all__ += funcs.__all__

# Cleanup
del _base, _composite, funcs, _galilean, _identity, _sequential
