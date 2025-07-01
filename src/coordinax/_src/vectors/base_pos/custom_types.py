"""Representation of coordinates in different systems."""

__all__ = ["PosT"]

from typing_extensions import TypeVar

from .core import AbstractPos

PosT = TypeVar("PosT", bound=AbstractPos, default=AbstractPos)
