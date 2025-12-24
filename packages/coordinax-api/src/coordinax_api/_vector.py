"""Vector API for coordinax.

Copyright (c) 2023 Coordinax Devs. All rights reserved.
"""

__all__ = ("vconvert",)

from typing import Any

import plum


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
