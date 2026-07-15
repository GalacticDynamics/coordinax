"""Galilean coordinate reflections."""

__all__ = ("Reflect",)


from jaxtyping import Array, Shaped
from typing import Any, Final, TypeAlias, final

import plum
from jax.typing import ArrayLike

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

from .base import AbstractTransform
from .identity import identity
from .linear import AbstractLinearTransform
from coordinax.transforms._src import groups

HMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_ZERO_NORMAL: Final = "Reflect.from_normal requires a nonzero normal vector."


@final
class Reflect(AbstractLinearTransform):
    r"""Operator for Euclidean hyperplane reflections.

    A reflection across the hyperplane orthogonal to a nonzero normal vector $n$
    acts on Cartesian coordinates by the Householder matrix

    $$ H_n = I - 2\hat{n}\hat{n}^T, $$

    where $ \hat{n} = n / \lVert n \rVert $.

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

    """

    H: HMatrix
    """The reflection matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.OrthogonalGroup, groups.DiffeomorphismGroup))

    def __init__(self, H: Any) -> None:
        object.__setattr__(self, "H", jnp.asarray(H))

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
        if bool(jnp.allclose(norm, 0)):
            raise ValueError(_MSG_ZERO_NORMAL)

        n_hat = n / norm
        H = jnp.eye(n.shape[0], dtype=n_hat.dtype) - 2 * jnp.outer(n_hat, n_hat)
        return cls(H)

    @property
    def inverse(self) -> "Reflect":
        """The inverse of a reflection is the reflection itself."""
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
def simplify(op: Reflect, /, **kw: Any) -> AbstractTransform:
    """Simplify a reflection, collapsing the identity matrix when present."""
    if jnp.allclose(op.H, jnp.eye(op.H.shape[0], dtype=op.H.dtype), **kw):
        return identity
    return op
