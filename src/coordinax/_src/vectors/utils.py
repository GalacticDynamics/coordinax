"""Representation of coordinates in different systems."""

__all__: list[str] = []

from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING

import quaxed.numpy as jnp
from dataclassish import field_values

if TYPE_CHECKING:
    from coordinax import AbstractVector


def full_shaped(obj: "AbstractVector", /) -> "AbstractVector":
    """Return the vector, fully broadcasting all components.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> v = cx.CartesianPos2D(Quantity([1], "m"), Quantity([3, 4], "m"))
    >>> v.x.shape
    (1,)
    >>> v.y.shape
    (2,)

    >>> from coordinax._src.vectors.utils import full_shaped
    >>> full_shaped(v).x.shape
    (2,)

    """
    arrays = jnp.broadcast_arrays(*field_values(obj))
    return _dataclass_replace(obj, **dict(zip(obj.components, arrays, strict=True)))
