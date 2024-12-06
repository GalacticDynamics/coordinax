"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = [
    "represent_as",
    "normalize_vector",
]

from typing import Any

from plum import dispatch


@dispatch.abstract  # type: ignore[misc]
def represent_as(current: Any, target: type[Any], /, **kwargs: Any) -> Any:
    """Transform the current vector to the target vector.

    See the dispatch implementations for more details.

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract  # type: ignore[misc]
def normalize_vector(x: Any, /) -> Any:
    """Return the unit vector."""
    raise NotImplementedError
