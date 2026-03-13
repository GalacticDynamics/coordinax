"""Representations."""

__all__ = ("vconvert", "guess_rep")

from typing import Any

import plum


@plum.dispatch.abstract
def vconvert(*args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target chart.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import coordinax.representations as cxr

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_rep(*args: Any, **kwargs: Any) -> Any:
    """Guess the representation of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_rep(data)
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    raise NotImplementedError  # pragma: no cover
