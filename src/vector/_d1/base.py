"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector", "Abstract1DVectorDifferential"]


from functools import partial

import equinox as eqx
import jax
from jax_quantity import Quantity

from vector._base import AbstractVector, AbstractVectorDifferential


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""

    @partial(jax.jit)
    def norm(self) -> Quantity["length"]:
        """Return the norm of the vector."""
        from .builtin import Cartesian1DVector  # pylint: disable=C0415

        return self.represent_as(Cartesian1DVector).norm()


class Abstract1DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 1D differentials in different systems."""

    vector_cls: eqx.AbstractClassVar[type[Abstract1DVector]]

    @partial(jax.jit)
    def norm(self, position: Abstract1DVector, /) -> Quantity["length"]:
        """Return the norm of the vector."""
        from .builtin import CartesianDifferential1D  # pylint: disable=C0415

        return self.represent_as(CartesianDifferential1D, position).norm()
