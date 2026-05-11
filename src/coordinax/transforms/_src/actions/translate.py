"""Translation operator."""

__all__ = ("Translate",)


from collections.abc import Callable
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

    """
    del kw

    if rep != cxr.point:
        raise TypeError("Translate can only be applied to point representations")

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


def _require_tau_time(tau: Any, /) -> tuple[Array, u.AbstractUnit]:
    """Return (tau_value, tau_unit), requiring tau to be a time quantity."""
    # We require a Quantity-like tau with time dimension so we can form d/dtau.
    tau = eqx.error_if(
        tau,
        not u.quantity.is_any_quantity(tau),
        "Translate requires tau as a time Quantity when delta is time-dependent",
    )
    tau_unit = u.unit_of(tau)
    tau = eqx.error_if(
        tau,
        u.dimension_of(tau_unit) != u.dimension_of(u.unit("s")),
        "Translate requires tau to have time dimension",
    )
    # Differentiate with respect to the numeric tau in its own unit.
    return jnp.asarray(u.ustrip(tau_unit, tau)), tau_unit  # ty: ignore[invalid-return-type]


def _delta_values_fn(
    delta_fn: Callable[[Any], Any],
    chart: cxc.AbstractChart,
    tau_unit: u.AbstractUnit,
    /,
) -> tuple[Callable[[Array], Array], u.AbstractUnit | None]:
    """Build a pure-JAX function returning packed delta values.

    Returns
    -------
    (f, unit)
        f: tau_value -> Array[... , n] of delta values in `chart.components` order.
        unit: common length unit returned by `pack_uniform_unit`.

    Notes
    -----
    We wrap `tau_value` back into a Quantity with unit `tau_unit` before calling
    the user-provided `delta_fn`.

    """
    keys = chart.components

    def f(tau_value: Array) -> Array:
        tau_q = u.Q(tau_value, tau_unit)
        d = delta_fn(tau_q)
        vals, _unit = pack_uniform_unit(d, keys=keys)
        # Ensure an array output for JAX differentiation.
        return jnp.asarray(vals)

    # One eager call to learn the unit.
    vals0, unit0 = pack_uniform_unit(delta_fn(u.Q(jnp.zeros(()), tau_unit)), keys=keys)
    del vals0

    return f, unit0


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
    >>> import unxt as u

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = u.Q([0.0, 0.0, 0.0], "m")
    >>> cxfm.act(shift, None, x, cxc.cart3d, cxr.point)
    Q([1000., 2000., 3000.], 'm')

    """
    if rep != cxr.point:
        raise TypeError("Translate can only be applied to point representations")

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

    Translation is performed in the canonical Cartesian chart of ``chart``:

    1. Convert the point ``x`` to ``cart = chart.cartesian``.
    2. Convert ``delta`` to the same ``cart`` using
        ``api.phys_tangent_basis_change`` evaluated at the point being
        translated (expressed in ``op.chart``).
    3. Add in Cartesian components.
    4. Convert the translated point back to ``chart``.

    Examples
    --------
    >>> import coordinax.transforms as cxfm
    >>> import unxt as u

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxfm.act(shift, None, x, cxc.cart3d, cxr.point)
    {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}

    """
    del kw

    if rep != cxr.point:
        raise TypeError("Translate can only be applied to point representations")

    op_eval = materialize_transform(op, tau)

    # Shared canonical Cartesian chart for performing the translation.
    cart = chart.cartesian

    # Convert point to Cartesian.
    x_cart = cxc.pt_map(x, chart, cart, usys=usys)

    # Convert delta to Cartesian (as a physical displacement).
    if op_eval.chart == cart:
        delta_cart = op_eval.delta
    else:
        raise NotImplementedError(
            "Translation by a delta in a different chart is not implemented yet."
        )
        # # Base point expressed in the delta's chart.
        # at_in_op_chart = cxc.pt_map(x, chart, op_eval.chart, usys=usys)
        # delta_cart = cxtapi.phys_tangent_basis_change(
        #     op_eval.delta, op_eval.chart, cart, at=at_in_op_chart, usys=usys
        # )

    # Add in Cartesian components (key-wise).
    x_cart2 = jtu.map(
        jnp.add,
        *((x_cart, delta_cart) if op_eval.right_add else (delta_cart, x_cart)),
        is_leaf=u.quantity.is_any_quantity,
    )

    # Convert back to the input chart.
    out = cxc.pt_map(x_cart2, cart, chart, usys=usys)
    return cast("CDict", out)
