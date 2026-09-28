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
from .linear import AbstractLinearTransform, as_dimensionless_matrix
from .utils import _unnormalisable
from coordinax.transforms._src import groups

HMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_ZERO_NORMAL: Final = "Reflect.from_normal needs a finite, nonzero normal."
_MSG_NOT_A_REFLECTION: Final = (
    "Reflect requires a hyperplane reflection: H symmetric, H @ H = I, and "
    "trace H = n - 2. The first two give an involution that `inverse` can "
    "return unchanged; the trace pins exactly one -1 eigenvalue, i.e. "
    "H = I - 2 n n^T for a unit normal n. Matrices that satisfy only the "
    "first two are involutions but not reflections -- the identity "
    "(trace n), a rotation by pi about an axis such as diag(-1, -1, 1) "
    "(trace n - 4), the point inversion -I (trace -n) -- and det H = -1 "
    "does not separate them either, since -I has det -1 in odd dimensions. "
    "For an orthogonal map with det = +1 use `Rotate`; for any other "
    "invertible map use `Linear`, including a rotoreflection, which is "
    "orthogonal with det = -1 but not an involution."
)

_ATOL: Final = 1e-6
"""Absolute tolerance on the reflection invariants. See `_not_a_reflection`."""


def _not_a_reflection(H: Any, /) -> Any:
    r"""Whether ``H`` fails to be a hyperplane reflection.

    The exact characterisation is **symmetric, involutive, and
    ``trace H == n - 2``**. Symmetric-and-involutive makes ``H`` orthogonal
    with eigenvalues in :math:`\{+1, -1\}`; the trace is then :math:`n - 2k`
    for :math:`k` eigenvalues equal to :math:`-1`, so ``trace H == n - 2`` is
    exactly :math:`k = 1` -- one reflected direction, i.e.
    :math:`H = I - 2 \hat n \hat n^T`.

    Both extra clauses are load-bearing. ``det H == -1`` alone does not
    suffice: the point inversion :math:`-I` has ``det -1`` in odd dimensions
    and reflects every direction. Symmetry is not implied either:
    ``[[1, 1], [0, -1]]`` is involutive with ``det -1`` *and*
    ``trace == n - 2``, yet is neither symmetric nor orthogonal.

    A non-square ``H`` answers `False`: it has no square to compare, and
    `_validate_square` is the one that names a bad shape. The shape is static
    under tracing, so this branch traces.

    ``atol`` and ``rtol`` are both explicit, for the reasons spelled out on
    `Rotate`'s predicate. ``rtol=0`` matters more here: ``trace H`` is
    compared against ``n - 2``, so the default ``rtol=1e-5`` would scale the
    budget with the dimension -- ``8.1e-5`` at ``n=10``, ``9.8e-4`` at
    ``n=100`` -- rather than holding the stated ``1e-6``.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return False
    n = H.shape[0]
    sq = jnp.matmul(H, H)
    return (
        ~jnp.allclose(H, H.T, atol=_ATOL, rtol=0.0)
        | ~jnp.allclose(sq, jnp.eye(n, dtype=sq.dtype), atol=_ATOL, rtol=0.0)
        | ~jnp.isclose(jnp.trace(H), n - 2, atol=_ATOL, rtol=0.0)
    )


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
        If ``H`` is not a hyperplane reflection -- symmetric, ``H @ H = I``,
        and ``trace H = n - 2``. An involution alone is not enough: the
        identity, a rotation by pi about an axis, and the point inversion
        ``-I`` are all involutions and all refused. These depend on the
        values, so the check is deferred onto the stored ``H`` to survive
        `jax.jit`: eagerly it raises from the constructor, under `jit` when
        the traced graph runs.

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

    A matrix that is not a hyperplane reflection is refused, and points at
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
        # Shape first: `_not_a_reflection` declines on a non-square matrix, so
        # without this one would be stored and `.inverse` would still hand back
        # `self`, which is undefined for a non-square `H`.
        H = self._validate_square(H)
        object.__setattr__(
            self, "H", eqx.error_if(H, _not_a_reflection(H), _MSG_NOT_A_REFLECTION)
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
    """Return the reflection unchanged: there is nothing to collapse.

    This used to collapse an identity-valued ``H`` to `Identity`. A `Reflect`
    can no longer *be* the identity: the constructor requires
    ``trace H == n - 2`` (exactly one reflected direction) and the identity has
    ``trace n``, so that branch became unreachable when the type was narrowed
    to hyperplane reflections. Dropping it also removes the only value
    inspection here, so this rule is trace-safe by construction rather than by
    an `is_traced` guard.
    """
    del approx, kw  # no value inspection left to switch off
    return op
