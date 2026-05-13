"""Translation operator."""

__all__ = ("Translate",)


from jaxtyping import Array, ArrayLike
from typing import Any, Union, cast, final

import equinox as eqx
import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
from .add import AbstractAdd
from .base import materialize_transform
from .composed import Composed
from .custom_types import CDict, OptUSys
from coordinax.internal import pack_uniform_unit
from coordinax.transforms._src import groups


@final
class Translate(AbstractAdd):
    r"""Operator for translating points.

    A Translate operator represents addition of a constant displacement $\Delta$
    in the ambient Euclidean space (or in a chart whose metric is Euclidean and
    whose canonical Cartesian chart exists).

    Think of $\Delta$ as a displacement vector field that is constant in space
    and time (unless explicitly time-dependent).

    Formally, in a Cartesian chart on $\mathbb{R}^n$: $T_\Delta:\; x \mapsto
    x+\Delta$.

    Its differential (pushforward) is the identity: $(dT_\Delta)_x = I$.

    Parameters
    ----------
    delta : CDict | Callable[[tau], CDict]
        The displacement to apply. Must have length dimension.  If callable,
        will be evaluated at the time parameter ``tau``.

    Notes
    -----
    - Only applicable to ``Point`` role vectors
    - Raises ``TypeError`` when applied to other roles (e.g., ``PhysVel``)
    - The ``delta`` is interpreted as a ``PhysDisp``-role displacement
      internally

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.transforms as cxfm
    >>> import wadler_lindig as wl

    Create a translation operator:

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> shift
    Translate(
        {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D(M=Rn(3))
    )

    The inverse negates the displacement:

    >>> shift.inverse
    Translate(
        {'x': Q(-1, 'km'), 'y': Q(-2, 'km'), 'z': Q(-3, 'km')}, chart=Cart3D(M=Rn(3))
    )

    Time-dependent translation:

    >>> delta = lambda t: {"x": u.Q(t.ustrip("s"), "m"), "y": u.Q(0, "m"),
    ...                    "z": u.Q(0, "m")}
    >>> moving = cxfm.Translate(delta, chart=cxc.cart3d)
    >>> moving
    Translate(<function <lambda>>, chart=Cart3D(M=Rn(3)))

    >>> t = u.Q(10, "s")
    >>> x = cx.cdict(u.Q([0, 0, 0], "m"))
    >>> wl.pprint(moving(t, x), short_arrays='compact', named_units=False)
    {'x': Quantity(10, unit='m'), 'y': Quantity(0, unit='m'),
     'z': Quantity(0, unit='m')}

    """

    semantic_kind: cxr.AbstractTangentSemanticKind = eqx.field(
        static=True, default=cxr.dpl
    )
    """Semantic kind of tangent data this operator acts on. Default: Displacement."""

    # delta, chart, and right_add inherited from AbstractAdd
    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.EuclideanGroup, groups.DiffeomorphismGroup))

    def __add__(self, other: object, /) -> Union["Translate", Composed]:
        """Combine two Translate operators with matching semantic kinds.

        Returns a combined Translate when both operators have the same
        ``semantic_kind``.  Returns a ``Composed`` when semantic kinds differ
        (since they act on different data and cannot be collapsed).

        """
        if not isinstance(other, Translate):
            return NotImplemented
        if self.semantic_kind != other.semantic_kind:
            return Composed((self, other))
        return super().__add__(other)  # ty: ignore[invalid-return-type]

    # inverse and __neg__ inherited from AbstractAdd


# ============================================================================
# act


@plum.dispatch
def act(
    op: Translate,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Translate to an ArrayLike.

    The array is interpreted as Cartesian coordinates. The delta is converted
    to the same unit system to perform the addition.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.transforms as cxfm

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> usys = u.unitsystems.si
    >>> cxfm.act(shift, None, x,  cxc.cart3d, cxr.point, usys=usys)
    Array([1000., 2000., 3000.], dtype=float64)

    Velocity-semantic translate is identity on point arrays:

    >>> import coordinax.representations as cxr
    >>> from dataclassish import replace
    >>> vel_shift = replace(shift, semantic_kind=cxr.vel)
    >>> cxfm.act(vel_shift, None, x, cxc.cart3d, cxr.point, usys=usys)
    Array([0., 0., 0.], dtype=float64)

    """
    del kw

    if rep != cxr.point:
        raise TypeError("Translate can only be applied to point representations")

    # A vel/acc-semantic translate does not move position points.
    if not isinstance(op.semantic_kind, cxr.Displacement):
        return jnp.asarray(x)

    if usys is None:
        raise TypeError("Translate requires usys to convert delta to x's units")

    chart = eqx.error_if(
        chart, not isinstance(chart, type(chart.cartesian)), "chart must be cartesian"
    )

    # Process Translation
    op_eval = materialize_transform(op, tau)

    # Convert delta to array using chart components and usys
    delta, unit = pack_uniform_unit(op_eval.delta, chart.components)  # ty: ignore[no-matching-overload]
    if unit is not None:
        delta = u.uconvert_value(usys[u.dimension_of(unit)], unit, delta)

    # Apply translation
    x_arr = jnp.asarray(x)
    return x_arr + delta if op_eval.right_add else delta + x_arr  # ty: ignore[unsupported-operator]


# -----------------------------------------------
# Special dispatches for Quantity.
# These are interpreted as Cartesian coordinates in a Euclidean metric
# The role is inferred from the dimensions.


@plum.dispatch
def act(
    op: Translate,
    tau: Any,
    x: u.AbstractQuantity,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Translate to a Quantity.

    The array is interpreted as Cartesian coordinates. The delta is converted
    to the same unit system to perform the addition.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr
    >>> from dataclassish import replace
    >>> import unxt as u

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = u.Q([0.0, 0.0, 0.0], "m")
    >>> cxfm.act(shift, None, x, cxc.cart3d, cxr.point)
    Q([1000., 2000., 3000.], 'm')

    Velocity-semantic translate is identity on point quantities:

    >>> vel_shift = replace(shift, semantic_kind=cxr.vel)
    >>> cxfm.act(vel_shift, None, x, cxc.cart3d, cxr.point)
    Q([0., 0., 0.], 'm')

    """
    if rep != cxr.point:
        raise TypeError("Translate can only be applied to point representations")

    # A vel/acc-semantic translate does not move position points.
    if not isinstance(op.semantic_kind, cxr.Displacement):
        return x

    # Process Translation
    op_eval = materialize_transform(op, tau)

    # Convert delta to array using chart components and usys
    delta = jnp.stack(list(op_eval.delta.values()), axis=-1)  # ty: ignore[unresolved-attribute]

    return x + delta if op_eval.right_add else delta + x


# -----------------------------------------------
# On CDict


@plum.dispatch
def act(
    op: Translate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply Translate to a Point-valued component dictionary.

    Dispatches on ``op.semantic_kind`` to determine which representations
    are shifted.

    Examples
    --------
    >>> import coordinax.transforms as cxfm
    >>> import unxt as u

    Default (displacement-semantic) translate shifts points:

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxfm.act(shift, None, x, cxc.cart3d, cxr.point)
    {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}

    """
    del kw
    return _act_translate_cdict(op.semantic_kind, op, tau, x, chart, rep, usys=usys)


@plum.dispatch
def _act_translate_cdict(
    sk: cxr.Displacement,
    op: Translate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Displacement-semantic translate: shifts points and displacement vectors."""
    op_eval = materialize_transform(op, tau)

    # A vel/acc-semantic translate does not move position points.
    # This check should never trigger here since dispatch selects on Displacement,
    # but explicit for consistency with other overloads.
    if rep == cxr.point and not isinstance(op.semantic_kind, cxr.Displacement):
        return x

    if rep == cxr.point:
        # Translate in Cartesian space, then map back.
        cart = chart.cartesian
        x_cart = cxc.pt_map(x, chart, cart, usys=usys)

        if op_eval.chart == cart:
            delta_cart = op_eval.delta
        else:
            # Push delta through the Jacobian into Cartesian.
            at_in_op_chart = cxc.pt_map(x_cart, cart, op_eval.chart, usys=usys)
            delta_cart = cxr.tangent_map(  # ty: ignore[missing-argument]
                op_eval.delta,
                op_eval.chart,
                cxr.coord_disp,
                cart,
                at=at_in_op_chart,
                usys=usys,
            )

        x_cart2 = jtu.map(
            jnp.add,
            *((x_cart, delta_cart) if op_eval.right_add else (delta_cart, x_cart)),
            is_leaf=u.quantity.is_any_quantity,
        )
        return cast("CDict", cxc.pt_map(x_cart2, cart, chart, usys=usys))

    if isinstance(rep.semantic_kind, cxr.Displacement):
        # Shift displacement-valued tangent representations.
        if op_eval.chart != chart:
            msg = (
                f"Translate.delta is defined in chart {op_eval.chart!r}, "
                f"but the representation is in chart {chart!r}. "
                "Convert delta to the target chart before constructing Translate."
            )
            raise ValueError(msg)
        return cast(
            "CDict",
            jtu.map(
                jnp.add,
                *((x, op_eval.delta) if op_eval.right_add else (op_eval.delta, x)),
                is_leaf=u.quantity.is_any_quantity,
            ),
        )

    # Identity for velocity, acceleration, and other representations.
    return x


@plum.dispatch
def _act_translate_cdict(
    sk: cxr.AbstractTangentSemanticKind,
    op: Translate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Velocity/acceleration-semantic translate: shifts matching tangent vectors."""
    op_eval = materialize_transform(op, tau)

    # A vel/acc-semantic translate does not move position points.
    if rep == cxr.point:
        return x

    if rep.semantic_kind == sk:
        if op_eval.chart != chart:
            msg = (
                f"Translate.delta is defined in chart {op_eval.chart!r}, "
                f"but the representation is in chart {chart!r}. "
                "Convert delta to the target chart before constructing Translate."
            )
            raise ValueError(msg)
        return cast(
            "CDict",
            jtu.map(
                jnp.add,
                *((x, op_eval.delta) if op_eval.right_add else (op_eval.delta, x)),
                is_leaf=u.quantity.is_any_quantity,
            ),
        )

    # Identity for non-matching representations.
    return x
