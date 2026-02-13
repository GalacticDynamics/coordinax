"""Vector API for coordinax."""

__all__ = ("vconvert", "cdict")

from typing import Any

import plum

from ._custom_types import CsDict


@plum.dispatch.abstract
def vconvert(target: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target representation.

    This is an abstract API definition. See the main coordinax package
    for concrete implementations and usage examples.

    Parameters
    ----------
    target : AbstractChart
        Target chart (representation) instance (not class) to convert to.
    *args
        Additional positional arguments (e.g., source representation, data).
    **kwargs
        Additional keyword arguments (e.g., units).

    Returns
    -------
    Any
        Converted representation data.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def cdict(obj: Any, /) -> CsDict:
    """Extract component dictionary from an object.

    This function converts various coordinate charts into a component
    dictionary where keys are component names and values are the corresponding
    values.

    Parameters
    ----------
    obj
        An object to extract a component dictionary from. Supported types include:
        - Vector: extracted from ``obj.data``
        - unxt.Quantity: treated as Cartesian coordinates with components in the
          last dimension, matched to the appropriate Cartesian chart
        - Mappings: returned as-is

    Returns
    -------
    dict[str, Any]
        A mapping from component names to values.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Extract from a Vector:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    Extract from a Quantity treated as Cartesian:

    >>> q = u.Q([1, 2, 3], "m")
    >>> d = cx.cdict(q)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    raise NotImplementedError  # pragma: no cover
