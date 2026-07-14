"""Translation operator."""

__all__ = ("Translate",)


from jaxtyping import Array, ArrayLike
from typing import Any, Union, cast, final

import equinox as eqx
import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
from .add import AbstractAdd
from .base import is_time_dependent, materialize_transform
from .composed import Composed
from .custom_types import CDict, OptUSys
from .prolong import prolong_slot, tau_derivative
from .utils import is_flat_chart
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

    In that flat setting its differential (pushforward) is the identity,
    $(dT_\Delta)_x = I$. When ``delta`` lives in a non-flat chart, or acts on
    data in a different or non-flat chart, the point action is base-point
    dependent and the differential is NOT the identity — see the Notes.

    Parameters
    ----------
    delta : CDict | Callable[[tau], CDict]
        The offset to apply. Its physical dimension follows ``semantic_kind``:
        length for the default displacement kind (``dpl``), speed for a
        velocity kick (``vel``), and so on up the time-derivative ladder. If
        callable, it is evaluated at the time parameter ``tau``.

    Notes
    -----
    The ``semantic_kind`` field sets the ladder order $k$ of the offset
    (``dpl``: $k=0$, ``vel``: $k=1$, ...). Acting on data of ladder order $m$
    (points behave as the curve position for $k=0$):

    - $k = 0$ with ``delta`` in a Cartesian-type (flat) chart: shifts points
      by $\delta(\tau)$; velocities gain $\dot\delta(\tau)$ and
      accelerations $\ddot\delta(\tau)$ when ``delta`` is time-dependent
      (the kinematic prolongation); ``Displacement`` data ($m = 0$) is
      unaffected (a displacement is a same-$\tau$ point difference and the
      Jacobian of a flat translation is the identity).
    - $k = 0$ with ``delta`` in a non-flat chart: the point action pushes
      ``delta`` through the chart Jacobian at the point, so it is base-point
      dependent. All tangent data — including displacements — transforms by
      the generic pushforward/prolongation of the point action, which is
      generally not the identity and requires the base point (``at=``, or a
      `~coordinax.Coordinate` bundle).
    - $k \geq 1$: identity on points and on all orders $m < k$; order $m = k$
      gains $\delta(\tau)$; orders $m > k$ gain
      $d^{m-k}\delta/d\tau^{m-k}$ when ``delta`` is time-dependent. The
      componentwise rule is definitional (the point action is the identity),
      independent of chart flatness.

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


_MSG_TAU_REQUIRED = (
    "act(Translate, ...) with a time-dependent (callable) delta requires a "
    "time parameter; got tau=None."
)


def _check_tau(op: "Translate", tau: Any, /) -> None:
    """Raise an informative TypeError before materializing a callable delta."""
    if tau is None and callable(op.delta):
        raise TypeError(_MSG_TAU_REQUIRED)


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
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Translate to an ArrayLike.

    The array is interpreted as Cartesian coordinates. The delta is converted
    to the same unit system to perform the addition.

    >>> import jax.numpy as jnp
    >>> import coordinax.transforms as cxfm

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> usys = u.unitsystems.si
    >>> cxfm.act(shift, None, x,  cxc.cart3d, cxr.point, usys=usys)
    Array([1000., 2000., 3000.], dtype=float64)

    A fibre kick (e.g. ``semantic_kind=vel``) cannot infer whether a bare,
    unitless array is a position (kick is identity) or the matching tangent
    data (kick applies) — that ambiguity is rejected loudly:

    >>> import coordinax.representations as cxr
    >>> from dataclassish import replace
    >>> vel_shift = replace(shift, semantic_kind=cxr.vel)
    >>> try:
    ...     cxfm.act(vel_shift, None, x, cxc.cart3d, cxr.point, usys=usys)
    ... except TypeError as e:
    ...     print(str(e)[:44])
    A fibre offset (Translate with semantic_kind

    """
    del kw

    if rep != cxr.point:
        raise TypeError("Translate can only be applied to point representations")

    # A bare, unitless array is ambiguous under a fibre kick: it could be a
    # position (kick is identity) or the kick's own tangent data (kick
    # applies). Silently choosing 'position' would drop the kick for velocity
    # arrays — reject instead. (Quantities are unambiguous: units select the
    # representation.)
    if not isinstance(op.semantic_kind, cxr.Displacement):
        msg = (
            f"A fibre offset (Translate with semantic_kind="
            f"{op.semantic_kind!r}) cannot act on a bare array: it is "
            "ambiguous whether the array is a position (offset is identity) "
            "or tangent data (offset applies). Use a Quantity with units, a "
            "component dict, or a typed vector."
        )
        raise TypeError(msg)

    if usys is None:
        raise TypeError("Translate requires usys to convert delta to x's units")

    chart = eqx.error_if(
        chart, not isinstance(chart, type(chart.cartesian)), "chart must be cartesian"
    )

    # Process Translation
    _check_tau(op, tau)
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
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Translate to a Quantity.

    The array is interpreted as Cartesian coordinates. The delta is converted
    to the same unit system to perform the addition.

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
    _check_tau(op, tau)
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
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    r"""Apply Translate to a component dictionary (ladder rule).

    The behavior follows the time-derivative ladder: with $k$ the operator's
    ``semantic_kind`` order and $m$ the input's ladder order, the input gains
    $d^{m-k}\delta/d\tau^{m-k}$ for $m \geq k$ (points act as the curve
    position for $k = 0$), and is unaffected for $m < k$ or for
    ``Displacement`` data ($m = 0$).

    >>> import coordinax.transforms as cxfm
    >>> import unxt as u

    Default (displacement-semantic) translate shifts points:

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> x = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxfm.act(shift, None, x, cxc.cart3d, cxr.point)
    {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}

    A static translate does not affect velocities:

    >>> v = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> cxfm.act(shift, None, v, cxc.cart3d, cxr.coord_vel)
    {'x': Q(1., 'km / s'), 'y': Q(0., 'km / s'), 'z': Q(0., 'km / s')}

    But a time-dependent translate boosts velocities by its rate
    (the kinematic prolongation):

    >>> delta = lambda t: {"x": u.Q(3.0, "km/s") * t, "y": u.Q(0.0, "km"),
    ...                    "z": u.Q(0.0, "km")}
    >>> moving = cxfm.Translate(delta, chart=cxc.cart3d)
    >>> cxfm.act(moving, u.Q(2.0, "s"), v, cxc.cart3d, cxr.coord_vel)
    {'x': Q(4., 'km / s'), 'y': Q(0., 'km / s'), 'z': Q(0., 'km / s')}

    """
    k = op.semantic_kind.order

    # --- Point input: the curve position, shifted only by a k=0 translate.
    if rep == cxr.point:
        if k != 0:
            return x
        return _translate_point_cdict(op, tau, x, chart, usys=usys)

    # --- Tangent input of ladder order m (int for all tangent kinds).
    m = cast("int", rep.semantic_kind.order)
    # Lower-order fibres are untouched by a higher-order offset.
    if m < k:
        return x

    # The componentwise ladder rule below is the prolongation of the point
    # action only when the k=0 delta and the data live in the SAME
    # Cartesian-type chart, where the point action is a true translation. If
    # delta lives in a non-flat chart (pushed through the chart Jacobian AT
    # the point) or the data's chart is non-flat / different (the translation
    # is nonlinear in the data's coordinates), the pushforward/prolongation
    # gains base-point-dependent coupling terms — defer to the generic
    # autodiff engine (which requires the base point 'at').
    if k == 0 and not (chart == op.chart and is_flat_chart(chart)):
        return _act_translate_nonflat(op, tau, x, chart, rep, m, kw, usys)

    # Displacements are same-tau point differences (never gain dtau terms and
    # the Jacobian of a flat translation is the identity).
    if m == 0:
        return x

    # Contribution: d^(m-k) delta / dtau^(m-k).
    n = m - k
    if n == 0:
        _check_tau(op, tau)
        delta = materialize_transform(op, tau).delta
    elif callable(op.delta):
        if tau is None:
            raise TypeError(_MSG_TAU_REQUIRED)
        delta = tau_derivative(op.delta, tau, n=n)
    else:
        # Static delta: all tau-derivatives vanish.
        return x

    # Only k >= 1 fibre kicks reach here cross-chart (k=0 routed to the
    # generic engine above). A kick is a tangent vector at the point, so it
    # has a well-defined cross-chart rule: push its components through the
    # chart Jacobian AT the base point (requires the anchor `at`).
    if op.chart != chart:
        delta = _kick_delta_in_chart(op, delta, chart, kw.get("at"), usys)

    return cast(
        "CDict",
        jtu.map(
            jnp.add,
            *((x, delta) if op.right_add else (delta, x)),
            is_leaf=u.quantity.is_any_quantity,
        ),
    )


def _kick_delta_in_chart(
    op: Translate,
    delta: CDict,
    chart: cxc.AbstractChart,
    at: CDict | None,
    usys: OptUSys,
    /,
) -> CDict:
    """Push a fibre-kick offset through the chart Jacobian at the base point."""
    if at is None:
        msg = (
            f"Translate.delta (a fibre offset) is defined in chart "
            f"{op.chart!r}, but the data is in chart {chart!r}. "
            "Converting the offset requires the base point: pass 'at' (a "
            "CDict in the data's chart) or use a coordinax.Coordinate "
            "bundle, which supplies it automatically."
        )
        raise TypeError(msg)
    at_in_op_chart = cxc.pt_map(at, chart, op.chart, usys=usys)
    kick_rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, op.semantic_kind)
    return cast(
        "CDict",
        cxr.tangent_map(  # ty: ignore[missing-argument]
            delta, op.chart, kick_rep, chart, at=at_in_op_chart, usys=usys
        ),
    )


def _act_translate_nonflat(
    op: Translate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    m: int,
    kw: dict[str, Any],
    usys: OptUSys,
    /,
) -> CDict:
    """Generic-engine fallback for a k=0 offset that is not a flat translation."""
    if m == 0 or not is_time_dependent(op):
        return cast(
            "CDict",
            cxfmapi.pushforward(op, tau, x, chart, rep, at=kw.get("at"), usys=usys),
        )
    return prolong_slot(
        op, tau, x, chart, m, at=kw.get("at"), at_vel=kw.get("at_vel"), usys=usys
    )


def _translate_point_cdict(
    op: Translate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Shift a point by the (materialized) delta, via the Cartesian chart."""
    _check_tau(op, tau)
    op_eval = materialize_transform(op, tau)

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
