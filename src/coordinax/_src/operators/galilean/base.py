# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__: list[str] = ["AbstractGalileanOperator"]


from abc import abstractmethod

from coordinax._src.operators.base import AbstractOperator


class AbstractGalileanOperator(AbstractOperator):
    """Abstract base class for Galilean operators on potentials."""

    @property
    @abstractmethod
    def is_inertial(self) -> bool: ...

    @property
    @abstractmethod
    def inverse(self) -> "AbstractOperator": ...
