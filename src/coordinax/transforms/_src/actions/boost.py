"""Boost (Galilean boost) operator."""

__all__ = ("Boost",)

from typing import Any, cast, final

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .add import AbstractAdd
from .builders import UniformTranslation
from .custom_types import CDict, OptUSys
from .timedep import TimeDep
from .utils import is_componentwise_offset
from coordinax.transforms._src.groups import AffineGroup, DiffeomorphismGroup

_MSG_TAU_REQUIRED_POINT = (
    "act(Boost, ...) on point data requires a time parameter: the Galilean "
    "boost moves points by delta_v * tau. Got tau=None."
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

    Equivalently, ``Boost(dv)`` is the uniform time-dependent translation
    ``TimeDep(UniformTranslation(dv))`` — the closed forms here are the
    prolongation of exactly that point action.

    Contrast with ``Translate(semantic_kind=vel)``: that operator is a pure
    velocity *kick* (an impulse) that shifts only the velocity fibre and does
    not move points.

    Parameters
    ----------
    delta : CDict
        The (constant) boost velocity.

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

    @property
    def is_time_dependent(self) -> bool:
        """Boost's point action is ``delta * tau`` — intrinsically tau-dependent.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.transforms as cxfm

        >>> dv = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
        >>> cxfm.Boost(dv, chart=cxc.cart3d).is_time_dependent
        True

        """
        return True


def _as_translate(op: Boost, /) -> TimeDep:
    r"""Return the equivalent displacement family: $\delta(\tau) = \Delta v\,\tau$."""
    return TimeDep(UniformTranslation(op.delta, chart=op.chart))


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
        # gating, and the generic fallback with anchors (at=, at_jet=).
        return cast(
            "CDict", cxfmapi.act(_as_translate(op), tau, x, chart, rep, usys=usys, **kw)
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

    # `dv` is constant, so higher derivatives of dv*tau vanish.
    if m != 1:
        return x
    return cast(
        "CDict",
        jtu.map(
            jnp.add,
            *((x, op.delta) if op.right_add else (op.delta, x)),
            is_leaf=u.quantity.is_any_quantity,
        ),
    )


@plum.dispatch
def act(
    op: Boost,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.TangentGeometry,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Boost tangent action (geometry-form): defer to the 5-arg ``act``.

    Boost's 5-arg ``act`` implements the kinematic prolongation directly (its
    closed forms avoid the general jet machinery); the geometry-form
    delegates to it so the two ``act`` forms are identical by construction.
    """
    del geom
    return cast("CDict", cxfmapi.act(op, tau, x, chart, rep, usys=usys, **kw))
