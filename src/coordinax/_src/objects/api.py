"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = ("normalize_vector",)

from typing import Any

import plum


@plum.dispatch.abstract
def normalize_vector(x: Any, /) -> Any:
    """Return the unit vector."""
    raise NotImplementedError  # pragma: no cover
