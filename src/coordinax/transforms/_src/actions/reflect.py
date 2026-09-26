"""Galilean coordinate reflections."""

__all__ = ("Reflect",)


from jaxtyping import Array, Shaped
from typing import Any, Final, TypeAlias, final

import equinox as eqx
import plum
from jax.typing import ArrayLike

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

from .base import AbstractTransform
from .identity import identity
from .linear import AbstractLinearTransform, as_dimensionless_matrix
from .utils import _unnormalisable, is_traced
from coordinax.transforms._src import groups

HMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_ZERO_NORMAL: Final = "Reflect.from_normal needs a finite, nonzero normal."
_MSG_NOT_INVOLUTIVE: Final = (
    "Reflect requires an involutive matrix: H @ H = I. That is the invariant "
    "`inverse` relies on -- it returns the operator itself. For an orthogonal "
    "map with det = +1 use `Rotate`; for any other invertible map use "
    "`Linear`. Note `Rotate` is SO(n), so an orthogonal matrix with "
    "det = -1 that is not an involution -- a rotoreflection -- belongs in "
    "`Linear`, not `Rotate`."
)

_ATOL: Final = 1e-6
"""Absolute tolerance on ``H @ H = I``. See `_not_involutive`."""


def _not_involutive(H: Any, /) -> Any:
    """Whether ``H`` fails ``H @ H = I``.

    A non-square ``H`` answers `False`: it has no square to compare, and
    `_validate_square` is the one that names a bad shape. The shape is static
    under tracing, so this branch traces.

    ``atol`` is explicit for the same reason as `Rotate`'s: the off-diagonal
    entries are compared against zero, so `jnp.allclose`'s ``1e-8`` is the
    whole budget and that is below the round-off of a numerically derived
    matrix.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return False
    sq = jnp.matmul(H, H)
    return ~jnp.allclose(sq, jnp.eye(H.shape[0], dtype=sq.dtype), atol=_ATOL)


@final
class Reflect(AbstractLinearTransform):
    r"""Operator for Euclidean hyperplane reflections.

    A reflection across the hyperplane orthogonal to a nonzero normal vector $n$
    acts on Cartesian coordinates by the Householder matrix

    $$ H_n = I - 2\hat{n}\hat{n}^T, $$

    where $ \hat{n} = n / \lVert n \rVert $.

    Raises
    ------
    equinox.EquinoxTracetimeError
        If ``H`` is not square. A shape is static, so this is decided while
        tracing and raises there -- not when the traced graph runs.
    equinox.EquinoxRuntimeError
        If ``H`` is not an involution. This one depends on the values, so it
        is deferred onto the stored ``H`` to survive `jax.jit`: eagerly it
        raises from the constructor, under `jit` when the traced graph runs.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Reflect.from_normal([1.0, 0.0, 0.0])
    >>> op.H
    Array([[-1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]], dtype=float64)

    >>> q = u.Q([1.0, 2.0, 3.0], "km")
    >>> cxfm.act(op, None, q)
    Q([-1.,  2.,  3.], 'km')

    A matrix that is not an involution is refused, and points at the types
    that fit:

    >>> P = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    >>> try:
    ...     cxfm.Reflect(P)
    ... except Exception as e:
    ...     print("H @ H = I" in str(e))
    True

    """

    H: HMatrix
    """The reflection matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.OrthogonalGroup, groups.DiffeomorphismGroup))

    def __init__(self, H: Any) -> None:
        # Involutivity, not orthogonality: `inverse` returns `self`, which is
        # right exactly when `H @ H = I`. (A *symmetric* involution is
        # orthogonal too, so for a Householder matrix this covers both; an
        # orthogonal matrix on its own does not imply it -- a permutation
        # matrix is orthogonal with `det = +1` and is not an involution.)
        #
        # Deferred so it survives jit (a plain `bool` on a traced value raises
        # `TracerBoolConversionError`), and threaded onto the stored array so
        # it is not dead-code-eliminated under trace.
        H = as_dimensionless_matrix(
            H,
            "Reflect `H` is a Householder matrix, whose entries are ratios "
            "and so dimensionless.",
        )
        # Shape first: `_not_involutive` declines on a non-square matrix, so
        # without this one would be stored and `.inverse` would still hand back
        # `self`, which is undefined for a non-square `H`.
        H = self._validate_square(H)
        object.__setattr__(
            self, "H", eqx.error_if(H, _not_involutive(H), _MSG_NOT_INVOLUTIVE)
        )

    @classmethod
    def from_normal(cls: type["Reflect"], normal: Any, /) -> "Reflect":
        """Construct a Householder reflection from a hyperplane normal."""
        n = jnp.asarray(normal)
        if n.ndim != 1:
            msg = (
                f"Reflect.from_normal requires a vector normal; got shape={n.shape!r}."
            )
            raise ValueError(msg)

        norm = jnp.linalg.norm(n)
        # Deferred so it survives jit (a plain `bool` on a traced value raises
        # TracerBoolConversionError). Anything else normalises to a NaN `H`.
        n = eqx.error_if(n, _unnormalisable(norm), _MSG_ZERO_NORMAL)

        n_hat = n / norm
        H = jnp.eye(n.shape[0], dtype=n_hat.dtype) - 2 * jnp.outer(n_hat, n_hat)
        return cls(H)

    @property
    def inverse(self) -> "Reflect":
        """The inverse of a reflection is the reflection itself.

        Nothing is checked here: `__init__` validates ``H @ H = I``, so an
        ``H`` that exists has already passed and this identity holds.
        """
        return self

    @property
    def _raw_matrix(self) -> Any:
        return self.H


@Reflect.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Reflect], obj: Reflect, /) -> Reflect:
    """Construct a Reflect from another Reflect."""
    return obj


@Reflect.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Reflect], obj: AbcQ, /) -> Reflect:
    """Construct a Reflect from a dimensionless quantity matrix."""
    return cls(u.ustrip("", obj))


@Reflect.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Reflect], obj: ArrayLike, /) -> Reflect:
    """Construct a Reflect from an array matrix."""
    return cls(obj)


@plum.dispatch
def simplify(op: Reflect, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Simplify a reflection, collapsing the identity matrix when present.

    The identity-matrix check inspects values, so it is skipped when
    ``approx=False``, and when the matrix is traced -- under `jax.jit` the
    values are not known, which is exactly when the answer is "do not
    simplify" rather than an error.
    """
    if (
        approx
        and not is_traced(op.H)
        and jnp.allclose(op.H, jnp.eye(op.H.shape[0], dtype=op.H.dtype), **kw)
    ):
        return identity
    return op
