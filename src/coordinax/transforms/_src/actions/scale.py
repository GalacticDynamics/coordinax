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
_MSG_NOT_DIAGONAL: Final = (
    "Scale requires a diagonal matrix -- it scales the axes, and nothing else. "
    "For a general linear map use `Linear`."
)


@final
class Scale(AbstractLinearTransform):
    r"""Operator for Cartesian linear scaling.

    A scaling transform applies

    $$
    x \mapsto Sx,
    $$

    where ``S`` is an invertible **diagonal** matrix: anisotropic scaling with
    one factor per axis.

    Diagonality is the whole content of the type. A `Scale` holding an
    off-diagonal matrix would be a general linear map wearing a name that
    promises otherwise, and `isinstance(op, Scale)` would stop meaning
    anything. `~coordinax.transforms.Linear` is the type for a general matrix.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.transforms as cxfm

    >>> jnp.diagonal(cxfm.Scale.from_factors(jnp.asarray([2.0, 3.0, 4.0])).matrix)
    Array([2., 3., 4.], dtype=float64)

    A matrix that is already diagonal is accepted:

    >>> jnp.diagonal(cxfm.Scale(jnp.diag(jnp.asarray([2.0, 1.0, 0.5]))).matrix)
    Array([2. , 1. , 0.5], dtype=float64)

    One that is not is refused, and points at the type that fits:

    >>> try:
    ...     cxfm.Scale(jnp.asarray([[1.0, 0.5], [0.0, 1.0]])).matrix
    ... except Exception as e:
    ...     print("Scale requires a diagonal matrix" in str(e))
    True

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
        """Return the inverse scaling transform.

        Reciprocals of the diagonal, not `jnp.linalg.inv`: O(n) rather than
        O(n^3), and exact where the general solve is not. Going through
        `matrix` rather than the raw field also means a malformed `S` is caught
        here with the same message as everywhere else, instead of surfacing as
        a raw LAPACK shape error.
        """
        return type(self)(jnp.diag(1.0 / jnp.diagonal(self.matrix)))

    @property
    def _raw_matrix(self) -> Any:
        return self._validate_diagonal(self.S)

    def _validate_diagonal(self, matrix: Any, /) -> Any:
        """Check the matrix has no off-diagonal entries.

        Deferred like the singular check so it survives `jit`: a plain `bool`
        on a traced value raises `TracerBoolConversionError`.
        """
        if matrix.ndim != 2:  # let the base's square check produce the message
            return matrix
        off = matrix - jnp.diag(jnp.diagonal(matrix))
        return eqx.error_if(matrix, jnp.any(off != 0), _MSG_NOT_DIAGONAL)


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


@plum.dispatch
def _merge(a: Scale, b: Scale, /) -> AbstractTransform | None:
    """Merge two adjacent scalings (``a`` applied first) into one, as ``b.S @ a.S``."""
    return Scale(b.S @ a.S)
