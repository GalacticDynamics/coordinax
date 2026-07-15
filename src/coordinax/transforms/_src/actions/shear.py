"""Pure spatial shear transform."""
# ruff: noqa: I001

__all__ = ("Shear",)


from typing import Any, TypeAlias, final

import plum
from jax.typing import ArrayLike
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

from .base import AbstractTransform
from .identity import identity
from .linear import AbstractLinearTransform
from coordinax.transforms._src import groups

HMatrix: TypeAlias = Shaped[Array, " N N"]


@final
class Shear(AbstractLinearTransform):
    r"""Operator for Cartesian linear shear.

    A shear transform applies

    $$
    x \mapsto Hx,
    $$

    where ``H`` is an invertible shear matrix.

    """

    H: HMatrix
    """The shear matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.AffineGroup, groups.DiffeomorphismGroup))

    def __init__(self, H: Any) -> None:
        object.__setattr__(self, "H", jnp.asarray(H))

    @property
    def inverse(self) -> "Shear":
        """Return the inverse shear transform."""
        return type(self)(jnp.linalg.inv(self.H))

    @property
    def _raw_matrix(self) -> Any:
        return self.H


@Shear.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Shear], obj: Shear, /) -> Shear:
    """Construct a Shear from another Shear."""
    return obj


@Shear.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Shear], obj: AbcQ, /) -> Shear:
    """Construct a Shear from a dimensionless quantity matrix."""
    return cls(u.ustrip("", obj))


@Shear.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Shear], obj: ArrayLike, /) -> Shear:
    """Construct a Shear from an array matrix."""
    return cls(obj)


@plum.dispatch
def simplify(op: Shear, /, **kw: Any) -> AbstractTransform:
    """Simplify a shear transform to identity when matrix is identity."""
    if jnp.allclose(op.H, jnp.eye(op.H.shape[0], dtype=op.H.dtype), **kw):
        return identity
    return op
