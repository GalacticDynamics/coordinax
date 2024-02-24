"""Representation of coordinates in different systems."""

__all__ = ["Abstract2DVector", "Abstract2DVectorDifferential"]


from functools import partial

import equinox as eqx
import jax
from jax_quantity import Quantity

from vector._base import AbstractVector, AbstractVectorDifferential


class Abstract2DVector(AbstractVector):
    """Abstract representation of 2D coordinates in different systems."""

    @partial(jax.jit)
    def norm(self) -> Quantity["length"]:
        """Return the norm of the vector."""
        from .builtin import Cartesian2DVector  # pylint: disable=C0415

        return self.represent_as(Cartesian2DVector).norm()


class Abstract2DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 2D vector differentials."""

    vector_cls: eqx.AbstractClassVar[type[Abstract2DVector]]

    @partial(jax.jit)
    def norm(self, position: Abstract2DVector, /) -> Quantity["length"]:
        """Return the norm of the vector."""
        from .builtin import CartesianDifferential2D  # pylint: disable=C0415

        return self.represent_as(CartesianDifferential2D, position).norm()
