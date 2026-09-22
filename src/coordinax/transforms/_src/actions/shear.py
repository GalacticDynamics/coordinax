"""Pure spatial shear transform."""
# ruff: noqa: I001

__all__ = ("Shear",)


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
from .scale import _singular
from .utils import is_traced
from coordinax.transforms._src import groups

HMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_SINGULAR: Final = (
    "Shear matrix must be invertible: det H finite, non-zero. That is the "
    "invariant `inverse` relies on -- it inverts `H`."
)


@final
class Shear(AbstractLinearTransform):
    r"""Operator for Cartesian linear shear.

    A shear transform applies

    $$
    x \mapsto Hx,
    $$

    where ``H`` is an invertible shear matrix.

    Raises
    ------
    equinox.EquinoxTracetimeError
        If ``H`` is not square. A shape is static, so this is decided while
        tracing and raises there -- not when the traced graph runs.
    equinox.EquinoxRuntimeError
        If ``H`` is singular. This one depends on the values, so it is
        deferred onto the stored ``H`` to survive `jax.jit`: eagerly it raises
        from the constructor, under `jit` when the traced graph runs.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Shear(jnp.asarray([[1.0, 0.5], [0.0, 1.0]]))
    >>> op.inverse.H
    Array([[ 1. , -0.5],
           [ 0. ,  1. ]], dtype=float64)

    A singular matrix is refused, rather than inverting to ``inf``/``nan``:

    >>> try:
    ...     cxfm.Shear(jnp.asarray([[1.0, 1.0], [1.0, 1.0]]))
    ... except Exception as e:
    ...     print("must be invertible" in str(e))
    True

    """

    H: HMatrix
    """The shear matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.AffineGroup, groups.DiffeomorphismGroup))

    def __init__(self, H: Any) -> None:
        # `Scale` had this same hole, fixed in #805; this is that guard on a
        # general matrix, so the predicate is reused with the determinant in
        # place of the diagonal factors.
        #
        # Deferred so it survives jit (a plain `bool` on a traced value raises
        # `TracerBoolConversionError`), and threaded onto the stored array so
        # it is not dead-code-eliminated under trace. Without it a singular
        # `H` reached `inverse` and came back all `inf`/`nan`.
        H = jnp.asarray(H)
        # Shape first: a non-square `H` has no determinant to take, so the
        # singularity check below declines on one and `inverse` would surface
        # a raw `jnp.linalg.inv` error instead of naming the shape.
        H = self._validate_square(H)
        bad = _singular(jnp.linalg.det(H))
        object.__setattr__(self, "H", eqx.error_if(H, bad, _MSG_SINGULAR))

    @property
    def inverse(self) -> "Shear":
        """Return the inverse shear transform.

        `__init__` has already established that ``H`` is invertible, so the
        inversion below is well posed; the new operator re-checks its own matrix
        on the way in, as any other construction would.
        """
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
def simplify(op: Shear, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Simplify a shear transform to identity when matrix is identity.

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
