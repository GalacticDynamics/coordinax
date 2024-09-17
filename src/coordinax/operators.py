"""Transformations of the coordinate reference frame.

E.g. a translation.
"""
# pylint: disable=unused-wildcard-import,wildcard-import

from ._src.operators import (
    base,
    composite,
    funcs,
    galilean,
    identity,
    sequential,
)
from ._src.operators.base import *  # noqa: F403
from ._src.operators.composite import *  # noqa: F403
from ._src.operators.funcs import *  # noqa: F403
from ._src.operators.galilean import *  # noqa: F403
from ._src.operators.identity import *  # noqa: F403
from ._src.operators.sequential import *  # noqa: F403

__all__: list[str] = []
__all__ += base.__all__
__all__ += composite.__all__
__all__ += sequential.__all__
__all__ += identity.__all__
__all__ += galilean.__all__
__all__ += funcs.__all__

# Cleanup
del base, composite, funcs, galilean, identity, sequential
