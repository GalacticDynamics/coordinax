"""Pure spatial scaling transform."""
# ruff: noqa: I001

__all__ = ("Scale",)


from typing import Any, Final, TypeAlias, final

import equinox as eqx
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

SMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_SINGULAR: Final = "Scale matrix must be invertible."


@final
class Scale(AbstractLinearTransform):
    r"""Operator for Cartesian linear scaling.

    A scaling transform applies

    $$
    x \mapsto Sx,
    $$

    where ``S`` is an invertible scaling matrix. The common case is diagonal
    anisotropic scaling with per-axis factors.

    """

    S: SMatrix
    """The scaling matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.AffineGroup, groups.DiffeomorphismGroup))

    def __init__(self, S: Any) -> None:
        object.__setattr__(self, "S", jnp.asarray(S))

    @classmethod
    def from_factors(cls: type["Scale"], factors: Any, /) -> "Scale":
        """Construct a diagonal scaling transform from axis factors."""
        s = jnp.asarray(factors)
        if s.ndim != 1:
            msg = f"Scale.from_factors requires a vector; got shape={s.shape!r}."
            raise ValueError(msg)
        # Defer the singular check so it survives jit (a plain `bool` on a
        # traced value raises TracerBoolConversionError).
        s = eqx.error_if(s, jnp.any(jnp.isclose(s, 0)), _MSG_SINGULAR)
        return cls(jnp.diag(s))

    @property
    def inverse(self) -> "Scale":
        """Return the inverse scaling transform."""
        return type(self)(jnp.linalg.inv(self.S))

    @property
    def _raw_matrix(self) -> Any:
        return self.S


@Scale.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Scale], obj: Scale, /) -> Scale:
    """Construct a Scale from another Scale."""
    return obj


@Scale.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Scale], obj: AbcQ, /) -> Scale:
    """Construct a Scale from a dimensionless quantity matrix."""
    return cls(u.ustrip("", obj))


@Scale.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Scale], obj: ArrayLike, /) -> Scale:
    """Construct a Scale from an array matrix."""
    return cls(obj)


@plum.dispatch
def simplify(op: Scale, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Simplify a scaling transform to identity when matrix is identity.

    The identity-matrix check inspects values, so it is skipped when
    ``approx=False``.
    """
    if approx and jnp.allclose(op.S, jnp.eye(op.S.shape[0], dtype=op.S.dtype), **kw):
        return identity
    return op
