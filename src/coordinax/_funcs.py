"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = ["represent_as"]

from typing import Any

from plum import dispatch


@dispatch.abstract  # type: ignore[misc]
def represent_as(current: Any, target: type[Any], /, **kwargs: Any) -> Any:
    """Transform the current vector to the target vector."""
    raise NotImplementedError  # pragma: no cover
