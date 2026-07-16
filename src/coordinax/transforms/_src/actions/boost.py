"""Boost (Galilean boost) operator."""

__all__ = ("Boost",)

from typing import Any, cast, final

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .add import AbstractAdd
from .custom_types import CDict, OptUSys
from .translate import Translate
from .utils import is_componentwise_offset
from coordinax.transforms._src.groups import AffineGroup, DiffeomorphismGroup

_MSG_TAU_REQUIRED_POINT = (
    "act(Boost, ...) on point data requires a time parameter: the Galilean "
    "boost moves points by delta_v * tau. Got tau=None."
)
_MSG_TAU_REQUIRED_TANGENT = (
    "act(Boost, ...) with a time-dependent delta on order-{m} tangent data "
    "requires a time parameter; got tau=None."
)


@final
class Boost(AbstractAdd):
    r"""Operator for Galilean boosts.

    A Galilean boost is the change to a frame moving at constant velocity
    $\Delta v$ (see the inhomogeneous Galilean group):

    $$ B_{\Delta v}:\; (\tau, x) \mapsto (\tau,\, x + \Delta v\, \tau). $$

    Its kinematic prolongation follows: points move by $\Delta v\,\tau$,
    velocities shift by $\Delta v$, and accelerations are unchanged (for a
    constant $\Delta v$). Displacements (same-$\tau$ point differences) are
    invariant.

    Equivalently, ``Boost(dv)`` is the time-dependent translation
    ``Translate(delta=lambda tau: dv * tau)`` — the closed forms here are the
    prolongation of exactly that point action.

    Contrast with ``Translate(semantic_kind=vel)``: that operator is a pure
    velocity *kick* (an impulse) that shifts only the velocity fibre and does
    not move points.

    Parameters
    ----------
    delta : CDict | Callable[[tau], CDict]
        The boost velocity. If callable, it is evaluated at the time
        parameter ``tau`` and the prolongation gains the corresponding
        derivative terms.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    Create a boost operator:

    >>> dv = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> boost = cxfm.Boost(dv, chart=cxc.cart3d)

    The boost moves points by ``dv * tau``:

    >>> p = {"x": u.Q(0.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(0.0, "km")}
    >>> cxfm.act(boost, u.Q(3.0, "s"), p, cxc.cart3d, cxr.point)
    {'x': Q(3., 'km'), 'y': Q(2., 'km'), 'z': Q(0., 'km')}

    and shifts velocities by ``dv``:

    >>> v = {"x": u.Q(2.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> cxfm.act(boost, u.Q(3.0, "s"), v, cxc.cart3d, cxr.coord_vel)
    {'x': Q(3., 'km / s'), 'y': Q(0., 'km / s'), 'z': Q(0., 'km / s')}

    The inverse negates the boost velocity:

    >>> boost.inverse.delta["x"]
    Q(-1., 'km / s')

    """

    # delta, chart, and right_add inherited from AbstractAdd
    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((AffineGroup, DiffeomorphismGroup))


def _boost_displacement(op: Boost, /) -> Any:
    r"""Return the boost's displacement function $g(\tau) = \Delta v(\tau)\tau$."""
    delta = op.delta

    def g(tau: Any, /) -> CDict:
        dv = delta(tau) if callable(delta) else delta  # ty: ignore[call-top-callable]
        return jtu.map(lambda c: c * tau, dv, is_leaf=is_any_quantity)

    return g


def _as_translate(op: Boost, /) -> Translate:
    """Return the equivalent displacement Translate: delta = dv(tau) * tau."""
    return Translate(_boost_displacement(op), chart=op.chart, right_add=op.right_add)


# ============================================================================
# act


@plum.dispatch
def act(
    op: Boost,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    r"""Apply a Galilean boost to a component dictionary.

    The point action is $x \mapsto x + \Delta v\,\tau$; tangent data of
    ladder order $m$ gains $d^m(\Delta v\,\tau)/d\tau^m$ (so $\Delta v$ for
    velocities and, for constant $\Delta v$, nothing for accelerations).
    Displacements are invariant.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> dv = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> boost = cxfm.Boost(dv, chart=cxc.cart3d)

    Boost shifts velocity components:

    >>> v = {"x": u.Q(2.0, "km/s"), "y": u.Q(3.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> cxfm.act(boost, u.Q(0.0, "s"), v, cxc.cart3d, cxr.coord_vel)
    {'x': Q(3., 'km / s'), 'y': Q(3., 'km / s'), 'z': Q(0., 'km / s')}

    A static boost leaves accelerations unchanged:

    >>> a = {"x": u.Q(1.0, "km/s2"), "y": u.Q(0.0, "km/s2"), "z": u.Q(0.0, "km/s2")}
    >>> cxfm.act(boost, u.Q(0.0, "s"), a, cxc.cart3d, cxr.coord_acc)
    {'x': Q(1., 'km / s2'), 'y': Q(0., 'km / s2'), 'z': Q(0., 'km / s2')}

    """

    def delegate() -> CDict:
        # The single delegation tail: the equivalent displacement Translate
        # (delta(tau) = dv*tau) implements the ladder rule, the flat-chart
        # gating, and the generic fallback with anchors (at=, at_vel=).
        return cast(
            "CDict",
            cxfmapi.act(_as_translate(op), tau, x, chart, rep, usys=usys, **kw),
        )

    # --- Point input: x + dv * tau, via the Translate ladder machinery.
    if rep == cxr.point:
        if tau is None:
            raise TypeError(_MSG_TAU_REQUIRED_POINT)
        return delegate()

    # The closed forms below hold only when dv and the data live in the same
    # Cartesian-type (flat) chart, where the boost's point action is a flat
    # translation at each tau. Otherwise the action is base-point dependent
    # in the data's coordinates — delegate everything (including
    # displacements).
    if not is_componentwise_offset(op, chart):
        return delegate()

    # --- Tangent input of ladder order m, flat matching chart.
    m = rep.semantic_kind.order
    # Displacements are invariant (the Jacobian of a flat translation is I).
    if m == 0:
        return x

    # Static dv: closed forms that need no time parameter.
    if not callable(op.delta):
        if m != 1:
            # Constant dv: higher derivatives of dv*tau vanish.
            return x
        return cast(
            "CDict",
            jtu.map(
                jnp.add,
                *((x, op.delta) if op.right_add else (op.delta, x)),
                is_leaf=u.quantity.is_any_quantity,
            ),
        )

    # Time-dependent dv: delegate to the equivalent displacement Translate,
    # whose ladder rule computes d^m (dv(tau) * tau) / dtau^m.
    if tau is None:
        raise TypeError(_MSG_TAU_REQUIRED_TANGENT.format(m=m))
    return delegate()
