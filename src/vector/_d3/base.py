"""Representation of coordinates in different systems."""

__all__ = ["Abstract3DVector", "Abstract3DVectorDifferential"]


from functools import partial

import equinox as eqx
import jax

from jax_quantity import Quantity

from vector._base import AbstractVector, AbstractVectorDifferential


class Abstract3DVector(AbstractVector):
    """Abstract representation of 3D coordinates in different systems."""

    @partial(jax.jit)
    def norm(self) -> Quantity["length"]:
        """Return the norm of the vector."""
        from .builtin import Cartesian3DVector  # pylint: disable=C0415

        return self.represent_as(Cartesian3DVector).norm()


class Abstract3DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 3D vector differentials."""

    vector_cls: eqx.AbstractClassVar[type[Abstract3DVector]]

    @partial(jax.jit)
    def norm(self, position: Abstract3DVector, /) -> Quantity["length"]:
        """Return the norm of the vector."""
        from .builtin import CartesianDifferential3D  # pylint: disable=C0415

        return self.represent_as(CartesianDifferential3D, position).norm()
