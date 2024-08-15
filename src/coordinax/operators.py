"""Transformations of the coordinate reference frame.

E.g. a translation.
"""
# pylint: disable=unused-wildcard-import,wildcard-import

from ._coordinax.operators import (
    _composite,
    _identity,
    base,
    funcs,
    galilean,
    sequential,
)
from ._coordinax.operators._composite import *  # noqa: F403
from ._coordinax.operators._identity import *  # noqa: F403
from ._coordinax.operators.base import *  # noqa: F403
from ._coordinax.operators.funcs import *  # noqa: F403
from ._coordinax.operators.galilean import *  # noqa: F403
from ._coordinax.operators.sequential import *  # noqa: F403

__all__: list[str] = []
__all__ += base.__all__
__all__ += _composite.__all__
__all__ += sequential.__all__
__all__ += _identity.__all__
__all__ += galilean.__all__
__all__ += funcs.__all__

# Cleanup
del base, _composite, funcs, galilean, _identity, sequential
