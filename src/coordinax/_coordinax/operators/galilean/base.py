# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__: list[str] = ["AbstractGalileanOperator"]


from abc import abstractmethod

from coordinax._coordinax.operators.base import AbstractOperator


class AbstractGalileanOperator(AbstractOperator):
    """Abstract base class for Galilean operators on potentials.

    A potential wrapper is a class that wraps another potential.
    The wrapped potential can be accessed through the `__wrapped__` attribute.
    """

    @property
    @abstractmethod
    def is_inertial(self) -> bool: ...

    @property
    @abstractmethod
    def inverse(self) -> "AbstractOperator": ...
