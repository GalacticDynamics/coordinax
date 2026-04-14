"""Representations."""

__all__: tuple[str, ...] = ("act", "compose", "frame_transition")

from typing import Any

import plum


@plum.dispatch.abstract
def act(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def compose(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def frame_transition(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def simplify(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError  # pragma: no cover
