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
SFactors: TypeAlias = Shaped[Array, " N"]

_MSG_SINGULAR: Final = "Scale matrix must be invertible: factors finite, non-zero."
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

    s: SFactors
    """The scaling factors -- the diagonal of $S$, one per axis.

    The diagonal rather than the whole matrix, so diagonality is structural
    instead of re-checked: a vector cannot have an off-diagonal entry. The
    matrix is rebuilt on demand by `_raw_matrix`, the same way
    `~coordinax.transforms.LorentzBoost` derives its own from a stored velocity.
    """

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.AffineGroup, groups.DiffeomorphismGroup))

    def __init__(self, S: Any) -> None:
        # Takes the matrix, stores its diagonal. Both checks run here, once,
        # rather than on every `matrix` access: what is stored cannot be
        # non-diagonal, so there is nothing left to re-check.
        #
        # This moves *when* a bad matrix is reported. Extracting the diagonal
        # consumes the checked value, so eagerly the error now surfaces at
        # construction rather than at the first `matrix` access. Under `jit` it
        # is deferred to runtime exactly as before, which is what the `error_if`
        # is for -- a plain `bool` on a tracer would raise at trace time.
        #
        # The `ndim` branch reads a static shape, so it survives tracing; a
        # non-2D input has no diagonal to take and carries the square error
        # forward in its place.
        S = self._validate_diagonal(self._validate_square(jnp.asarray(S)))
        object.__setattr__(self, "s", jnp.diagonal(S) if S.ndim == 2 else jnp.ravel(S))

    @classmethod
    def _from_diagonal(cls, s: Any, /) -> "Scale":
        """Build directly from factors, skipping the matrix round-trip.

        The diagonal is what the type stores, so every internal producer of one
        (`from_factors`, `inverse`, `_merge`) would otherwise inflate it to a
        matrix purely for `__init__` to take it apart again.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "s", jnp.asarray(s))
        return obj

    @classmethod
    def from_factors(cls: type["Scale"], factors: Any, /) -> "Scale":
        """Construct a diagonal scaling transform from axis factors."""
        s = jnp.asarray(factors)
        if s.ndim != 1:
            msg = f"Scale.from_factors requires a vector; got shape={s.shape!r}."
            raise ValueError(msg)
        # Deferred so it survives jit, as in `Reflect.from_normal`. `inf` is the
        # quiet failure: 1/inf = 0.0, so `inverse` came back finite and singular.
        bad = jnp.isclose(s, 0) | ~jnp.isfinite(s)
        s = eqx.error_if(s, jnp.any(bad), _MSG_SINGULAR)
        return cls._from_diagonal(s)

    @property
    def inverse(self) -> "Scale":
        """Return the inverse scaling transform.

        Reciprocals of the factors, not `jnp.linalg.inv`: O(n) rather than
        O(n^3), and exact where the general solve is not.

        Nothing is re-checked here, and `s` carries no deferred check of its
        own: `__init__` validates the matrix when it takes the diagonal, so an
        `s` that exists has already passed. A malformed input therefore reports
        from construction -- eagerly at the call, under `jit` when the traced
        graph runs -- never from this property.
        """
        return self._from_diagonal(1.0 / self.s)

    def _contract(self, matrix: Any, arr: Any, /) -> Any:
        """Scale each axis, rather than contracting a mostly-zero matrix.

        Takes the diagonal of the *validated* ``matrix`` rather than reading
        `s` directly: the shape checks ride along on that array as deferred
        `equinox.error_if` nodes, and reading the field instead would silently
        drop them.
        """
        return jnp.diagonal(matrix) * arr

    @property
    def _raw_matrix(self) -> Any:
        # No re-validation: `s` is a vector, so the matrix it builds is square
        # and diagonal by construction.
        return jnp.diag(self.s)

    def _validate_diagonal(self, matrix: Any, /) -> Any:
        """Check the matrix has no off-diagonal entries.

        Deferred like the singular check so it survives `jit`: a plain `bool`
        on a traced value raises `TracerBoolConversionError`.
        """
        # Non-square too, not just non-2D: `jnp.diag(jnp.diagonal(m))` on a
        # (3, 2) builds a (2, 2), and the subtraction below then dies with a
        # raw broadcasting error before the base can say "requires a square
        # matrix". Both shapes are the base's to report.
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
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
    if approx and jnp.allclose(op.s, 1.0, **kw):
        return identity
    return op


@plum.dispatch
def _merge(a: Scale, b: Scale, /) -> AbstractTransform | None:
    """Merge two adjacent scalings (``a`` applied first) into ``b.s * a.s``.

    Elementwise on the factors: composing two diagonal maps multiplies them
    axis by axis, so this is the $O(n)$ form of ``b.matrix @ a.matrix``. No
    validation here either -- both operands were checked when they were built,
    so a malformed one reports from its own construction and never reaches this.
    """
    return Scale._from_diagonal(b.s * a.s)
