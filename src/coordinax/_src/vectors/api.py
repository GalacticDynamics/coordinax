"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = [
    "vector",
    "vconvert",
    "normalize_vector",
]

from typing import Any

from plum import dispatch


@dispatch.abstract
def vconvert(target: type[Any], /, *args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target vector.

    See the dispatch implementations for more details. Not all transformations
    result in the target vector type, for example
    ``vconvert(type[Cartesian3DPos], FourVector)`` will return a
    `coordinax.vecs.FourVector` with the spatial part in Cartesian coordinates.
    Likewise, `coordinax.vconvert` on `coordinax.Coordinate` instances will
    transform the contained vectors to the target type, returning a
    `coordinax.Coordinate` instance.

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def normalize_vector(x: Any, /) -> Any:
    """Return the unit vector."""
    raise NotImplementedError


@dispatch.abstract
def vector(*args: Any, **kwargs: Any) -> Any:
    """Create a vector."""
    raise NotImplementedError  # pragma: no cover
