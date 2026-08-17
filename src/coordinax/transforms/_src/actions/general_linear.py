"""General invertible linear transform."""
# ruff: noqa: I001

__all__ = ("Linear",)


from typing import Any, TypeAlias, final

import equinox as eqx
import plum
from jax.typing import ArrayLike
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp

from .base import AbstractTransform
from .identity import identity
from .linear import AbstractLinearTransform
from coordinax.transforms._src import groups
from coordinax.transforms._src.groups import AbstractTransformGroup

LMatrix: TypeAlias = Shaped[Array, " N N"]


@final
class Linear(AbstractLinearTransform):
    r"""Operator for a general invertible Cartesian linear map.

    A linear transform applies

    $$
    x \mapsto Mx,
    $$

    where ``M`` is any invertible matrix.

    The other linear operators name a *structure* they preserve -- `Rotate` an
    orientation and a metric, `Reflect` a metric, `Scale` the axes. This one
    names none, which is what makes it the type a fused chain can land in:
    composing a rotation with a scaling leaves a matrix that is neither, and
    `simplify` needs somewhere to put it.

    Because it claims no structure of its own, it carries the group it *does*
    belong to as a field rather than declaring one for the class. Fusing two
    operators sets it to the least common supergroup of the pair, so a rotation
    composed with a reflection still reports `OrthogonalGroup` rather than
    falling all the way back to `AffineGroup`.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.transforms as cxfm

    A rotation and a scaling fuse into one operator:

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> S = cxfm.Scale.from_factors(jnp.asarray([2.0, 2.0, 2.0]))
    >>> fused = cxfm.simplify(R | S)
    >>> fused
    Linear(...)

    It agrees with applying the two in turn:

    >>> p = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> bool(jnp.allclose(fused(p)["y"].value, S(R(p))["y"].value))
    True

    The group is the least common supergroup of the pair, not the loosest one:

    >>> Rf = cxfm.Reflect.from_normal([0.0, 0.0, 1.0])
    >>> sorted(g.__name__ for g in cxfm.simplify(R | Rf).groups())
    ['DiffeomorphismGroup', 'OrthogonalGroup']

    """

    M: LMatrix
    """The transform matrix."""

    # Annotated with the bare name, not ``groups.AbstractTransformGroup``: the
    # ``groups`` method below shadows the module in the class namespace, and
    # Python 3.14 evaluates this annotation lazily, after that binding exists.
    group: type[AbstractTransformGroup] = eqx.field(
        static=True, default=groups.AffineGroup
    )
    """The most specific group this matrix is known to belong to.

    A field rather than a class-level declaration because the answer depends on
    what was fused, not on the type. Group classes are registered static, so
    this stays a pytree leaf-free constant.
    """

    def __init__(
        self, M: Any, group: type[AbstractTransformGroup] = groups.AffineGroup
    ) -> None:
        object.__setattr__(self, "M", jnp.asarray(M))
        object.__setattr__(self, "group", group)

    def groups(self) -> frozenset[type]:
        """Return the groups to which this map belongs.

        An instance method, not a classmethod: the group travels with the
        matrix. `~coordinax.transforms.Composed` does the same for the same
        reason.
        """
        return frozenset((self.group, groups.DiffeomorphismGroup))

    @property
    def inverse(self) -> "Linear":
        """Return the inverse linear transform.

        The group is carried across unchanged -- a group is closed under
        inverses, so whatever the matrix belonged to, its inverse does too.
        """
        return type(self)(jnp.linalg.inv(self.M), self.group)

    @property
    def _raw_matrix(self) -> Any:
        return self.M


@Linear.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Linear], obj: ArrayLike, /) -> Linear:
    """Construct a Linear from an array matrix.

    Registered on the class's own dispatcher, as the sibling transforms do.
    Via the global `plum.dispatch` the `ArrayLike` union is unfaithful, which
    would turn off plum's method cache for *every* `from_` in the library --
    caught by `test_no_new_unfaithful_signatures`.
    """
    return cls(obj)


@plum.dispatch
def simplify(op: Linear, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Simplify a general linear transform to identity when its matrix is one.

    The identity-matrix check inspects values, so it is skipped when
    ``approx=False``.
    """
    if approx and jnp.allclose(op.M, jnp.eye(op.M.shape[0], dtype=op.M.dtype), **kw):
        return identity
    return op


@plum.dispatch
def _merge(
    a: AbstractLinearTransform, b: AbstractLinearTransform, /
) -> AbstractTransform | None:
    """Fuse any two adjacent linear maps (``a`` first) into their matrix product.

    The generic fallback. Same-type rules -- `Rotate` with `Rotate`, `Scale`
    with `Scale` -- are more specific and so still win, keeping their tighter
    return type; this catches the mixed pairs that previously stayed a
    two-element `Composed` and paid one full ``act`` dispatch each.

    Returns `None` for operators of different dimension, such as a 3x3 `Rotate`
    beside a 4x4 `~coordinax.transforms.LorentzBoost`: there is no product to
    take. Shapes are static under `jax.jit`, so the check traces.
    """
    a_mat, b_mat = a.matrix, b.matrix
    if a_mat.shape != b_mat.shape:
        return None
    # `ty` cannot see `groups()`: every concrete transform defines it, but
    # `AbstractTransform` never declares it, so it is a protocol by convention
    # only. `Composed.groups` makes the same call and escapes the check merely
    # because its `transforms` field is loosely typed.
    group = groups.least_common_supergroup(
        (
            groups.most_specific_group(a.groups()),  # ty: ignore[unresolved-attribute]
            groups.most_specific_group(b.groups()),  # ty: ignore[unresolved-attribute]
        )
    )
    return Linear(b_mat @ a_mat, group)
