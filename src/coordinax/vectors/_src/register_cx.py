"""Register coordinax-related dispatches."""

__all__: tuple[str, ...] = ()

from dataclasses import replace

from typing import Any, cast

import jax.tree as jtu
import plum

import dataclassish
import quaxed.numpy as jnp
import unxt.quantity as uq

import coordinax.api.representations as cxrapi
import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .bundle import Coordinate
from .point import Point
from .tangent import Tangent
from coordinax.internal import CDict, OptUSys

CHART_MSMTCH = "from_chart {0} does not match the point's chart {1.chart}"

# ===================================================================
# Tangent conversion


@plum.dispatch
def cconvert(
    from_vec: Point, to_chart: cxc.AbstractChart, /, *, usys: OptUSys = None
) -> Point:
    """Convert a point from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.cconvert(vec, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call the `cconvert` function on the data from the vector's kind
    p = cxr.cconvert(from_vec.data, from_vec.chart, from_vec.rep, to_chart, usys=usys)
    # Return a new vector
    return replace(from_vec, data=p, chart=to_chart)


@plum.dispatch
def cconvert(
    from_vec: Point,
    from_chart: cxc.AbstractChart,
    to_chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Point:
    """Convert a vector from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> sph_vec = cx.cconvert(vec, cx.cart3d, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    if from_chart != from_vec.chart:
        raise ValueError(CHART_MSMTCH.format(from_chart, from_vec))

    out = cxr.cconvert(from_vec, to_chart, usys=usys)
    return cast("Point", out)


# -------------------------------------------------


@plum.dispatch
def pt_map(
    from_vec: Point, to_chart: cxc.AbstractChart, /, *, usys: OptUSys = None
) -> Point:
    """Convert a point from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cxc.pt_map(vec, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call `pt_map` on the data from the vector's kind
    p = cxc.pt_map(from_vec.data, from_vec.chart, from_vec.rep, to_chart, usys=usys)
    # Return a new vector
    return replace(from_vec, data=p, chart=to_chart)


@plum.dispatch
def pt_map(
    from_vec: Point,
    from_chart: cxc.AbstractChart,
    to_chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Point:
    """Convert a vector from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> sph_vec = cxc.pt_map(vec, cxc.cart3d, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    if from_chart != from_vec.chart:
        raise ValueError(CHART_MSMTCH.format(from_chart, from_vec))

    out = cxc.pt_map(from_vec, to_chart, usys=usys)
    return cast("Point", out)


# ===================================================================
# cdict dispatch


@plum.dispatch
def cdict(obj: Point, /) -> CDict:
    """Extract component dictionary from a Point.

    >>> import coordinax.main as cx
    >>> import unxt as u
    >>> vec = cx.Point.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    return obj.data


@plum.dispatch
def cdict(obj: Tangent, /) -> CDict:
    """Extract component dictionary from a Tangent.

    >>> import coordinax.main as cx
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> vec = cx.Tangent.from_(d, cxc.cart3d, cxr.coord_vel)
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    return obj.data


# ===================================================================
# Tangent cconvert


@plum.dispatch
def cconvert(
    from_vec: Tangent,
    to_chart: cxc.AbstractChart,
    /,
    *,
    at: Any = None,
    usys: OptUSys = None,
) -> Tangent:
    """Convert a tangent Tangent from one chart to another.

    The ``at`` parameter provides the base point at which the tangent map
    (Jacobian pushforward) is evaluated. It may be a `Point` instance
    (whose ``.data`` is used) or a raw ``CDict``.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_basis, cxr.vel,
    ... )
    >>> pt = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> v_sph = cx.cconvert(v, cxc.sph3d, at=pt)
    >>> v_sph.chart
    Spherical3D(M=Rn(3))

    """
    if from_vec.chart != to_chart:
        if at is None:
            msg = (
                f"'at' is required when converting a Tangent between different charts "
                f"({from_vec.chart!r} -> {to_chart!r}): "
                "the Jacobian pushforward needs a base point."
            )
            raise TypeError(msg)
        if isinstance(at, Point) and at.chart != from_vec.chart:
            msg = (
                f"'at' chart {at.chart!r} does not match "
                f"the source chart {from_vec.chart!r}."
            )
            raise ValueError(msg)
    at_data: CDict | None = at.data if isinstance(at, Point) else at
    p = cxr.cconvert(
        from_vec.data,
        from_vec.chart,
        from_vec.rep,
        to_chart,
        **({"at": at_data} if at_data is not None else {}),
        usys=usys,
    )
    return replace(from_vec, data=p, chart=to_chart)


@plum.dispatch
def cconvert(
    from_vec: Tangent,
    from_chart: cxc.AbstractChart,
    to_chart: cxc.AbstractChart,
    /,
    *,
    at: Any = None,
    usys: OptUSys = None,
) -> Tangent:
    """Convert a tangent Tangent from one chart to another (explicit from-chart).

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_basis, cxr.vel,
    ... )
    >>> pt = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> v_sph = cx.cconvert(v, cxc.cart3d, cxc.sph3d, at=pt)
    >>> v_sph.chart
    Spherical3D(M=Rn(3))

    """
    if from_chart != from_vec.chart:
        raise ValueError(CHART_MSMTCH.format(from_chart, from_vec))
    return cxrapi.cconvert(from_vec, to_chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


# ===================================================================
# Basis change


@plum.dispatch
def change_basis(
    v: Tangent,
    to_basis: cxr.AbstractLinearBasis,
    /,
    *,
    at: Any = None,
    usys: OptUSys = None,
) -> Tangent:
    """Change the basis of a `Tangent` vector.

    Converts the component data from the current basis to ``to_basis`` using the
    registered ``change_basis`` overload for dicts, then returns a new `Tangent`
    with the updated data and basis.

    The ``at`` parameter provides the base point at which the scale factors are
    evaluated.  It may be a `Point` instance (whose ``.data`` is used) or a raw
    ``CDict``.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    Convert a coordinate-basis spherical velocity to the physical basis:

    >>> point = cx.Point.from_(
    ...     {"r": u.Q(1.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.0, "rad")},
    ...     cxc.sph3d,
    ... )
    >>> vel = cx.Tangent.from_({"r": u.Q(1.0, "m/s"), "theta": u.Q(0.0, "rad/s"),
    ...     "phi": u.Q(0.0, "rad/s")}, cxc.sph3d, cxr.coord_vel)
    >>> vel_phys = cxr.change_basis(vel, cxr.phys_basis, at=point)
    >>> vel_phys.basis
    phys_basis
    >>> vel_phys.rep
    phys_vel

    """
    if v.basis != to_basis:
        if at is None:
            msg = (
                f"'at' is required when changing the basis of a Tangent "
                f"({v.basis!r} -> {to_basis!r}): "
                "scale factors need a base point."
            )
            raise TypeError(msg)
        if isinstance(at, Point) and at.chart != v.chart:
            msg = (
                f"'at' chart {at.chart!r} does not match "
                f"the vector's chart {v.chart!r}."
            )
            raise ValueError(msg)
    at_data: CDict | None = at.data if isinstance(at, Point) else at
    new_data = cxr.change_basis(
        v.data,
        v.chart,
        v.basis,
        to_basis,
        **({"at": at_data} if at_data is not None else {}),
        usys=usys,
    )
    return replace(v, data=new_data, basis=to_basis)


@plum.dispatch
def change_basis(
    v: Point, to_basis: cxr.AbstractLinearBasis, /, *, at: Any = None, usys: Any = None
) -> Tangent:
    """Promote a `Point` to a `Tangent` with `Displacement` semantics.

    The component data are unchanged; only the geometric interpretation is
    recast from a manifold point (`PointGeometry`) to a tangent-space
    displacement vector (`TangentGeometry`, `Displacement`).  The resulting
    `Tangent` carries the same chart and frame as the input `Point`, and its
    basis is ``to_basis``.

    The ``at`` and ``usys`` parameters are accepted for API consistency but are
    not used.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.representations as cxr

    >>> pt = cx.Point.from_([1.0, 2.0, 3.0], "m")
    >>> disp = cxr.change_basis(pt, cxr.coord_basis)
    >>> disp.semantic
    dpl
    >>> disp.basis
    coord_basis
    >>> disp.chart == pt.chart
    True

    """
    return Tangent(  # ty: ignore[missing-argument]
        data=v.data, chart=v.chart, basis=to_basis, semantic=cxr.dpl, frame=v.frame
    )


# ===================================================================
# Add / Subtract


@plum.dispatch
def add(lhs: Point, rhs: Point, /) -> Point:
    """Add two points.

    For non-Cartesian charts the operation converts both operands to the ambient
    Cartesian chart, adds there, and converts the result back to the ``lhs``
    chart.  For Cartesian charts the addition is direct.

    The result keeps the ``lhs`` chart and representation.

    >>> import coordinax.main as cx
    >>> v1 = cx.Point.from_([1, 2, 3], "m")
    >>> v2 = cx.Point.from_([4, 5, 6], "m")
    >>> print(cxr.add(v1, v2))
    <Point: chart=Cart3D (x, y, z) [m]
        [5 7 9]>

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot add vectors with different representations: "
            f"{lhs.rep} vs {rhs.rep}."
        )
        raise TypeError(msg)

    result_data = cxr.add(lhs.data, lhs.chart, lhs.rep, rhs.data, rhs.chart, rhs.rep)
    return replace(lhs, data=result_data)


@plum.dispatch
def subtract(lhs: Point, rhs: Point, /) -> Point:
    """Subtract two vectors.

    For non-Cartesian charts the operation converts both operands to the ambient
    Cartesian chart, subtracts there, and converts the result back to the
    ``lhs`` chart.  For Cartesian charts the subtraction is direct.

    The result keeps the ``lhs`` chart and representation.

    >>> import coordinax.main as cx
    >>> v1 = cx.Point.from_([4, 5, 6], "m")
    >>> v2 = cx.Point.from_([1, 2, 3], "m")
    >>> print(cxr.subtract(v1, v2))
    <Point: chart=Cart3D (x, y, z) [m]
        [3 3 3]>

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot subtract vectors with different representations: "
            f"{lhs.rep} vs {rhs.rep}."
        )
        raise TypeError(msg)

    result_data = cxr.subtract(
        lhs.data, lhs.chart, lhs.rep, rhs.data, rhs.chart, rhs.rep
    )
    return replace(lhs, data=result_data)


@plum.dispatch
def add(lhs: Tangent, rhs: Tangent, /) -> Tangent:
    """Add two tangent vectors component-wise.

    Tangent spaces are genuine vector spaces: addition is component-wise in any
    chart basis (no Cartesian round-trip is needed or correct).  Both operands
    must share the same representation (chart + basis + semantic).

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v1 = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> v2 = cx.Tangent.from_(
    ...     {"x": u.Q(4.0, "m/s"), "y": u.Q(5.0, "m/s"), "z": u.Q(6.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> result = cxr.add(v1, v2)
    >>> result["x"]
    Q(5., 'm / s')

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot add Tangent vectors with different representations: "
            f"{lhs.rep!r} vs {rhs.rep!r}."
        )
        raise ValueError(msg)

    data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


@plum.dispatch
def subtract(lhs: Tangent, rhs: Tangent, /) -> Tangent:
    """Subtract two tangent vectors component-wise.

    Tangent spaces are genuine vector spaces: subtraction is component-wise in
    any chart basis (no Cartesian round-trip is needed or correct).  Both
    operands must share the same representation (chart + basis + semantic).

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v1 = cx.Tangent.from_(
    ...     {"x": u.Q(4.0, "m/s"), "y": u.Q(5.0, "m/s"), "z": u.Q(6.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> v2 = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> result = cxr.subtract(v1, v2)
    >>> result["x"]
    Q(3., 'm / s')

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot subtract Tangent vectors with different representations: "
            f"{lhs.rep!r} vs {rhs.rep!r}."
        )
        raise ValueError(msg)

    data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


# ===================================================================
# `coordinax.representations`


@plum.dispatch
def act(
    op: cxfm.AbstractTransform,
    tau: Any,
    x: Tangent,
    /,
    *,
    at: Any = None,
    at_vel: Any = None,
    **kw: Any,
) -> Tangent:
    """Act a frame transform on a tangent Tangent.

    ``at`` (the base point) and ``at_vel`` (the velocity at the base point)
    anchor the transformation when it is needed — for Jacobian pushforwards in
    non-Cartesian charts and for the kinematic prolongation under
    time-dependent transforms. They may be `Point`/`Tangent` instances (whose
    ``.data`` is used, after a chart check) or raw ``CDict`` data.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cx.Rotate(Rz)
    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cx.cart3d, cx.coord_vel,
    ... )
    >>> transformed = cx.act(op, None, v)
    >>> print(transformed)
    <Tangent: chart=Cart3D (x, y, z) [m / s]
        [0. 1. 0.]>

    """
    at_data = _unwrap_anchor(at, x.chart, "at")
    at_vel_data = _unwrap_anchor(at_vel, x.chart, "at_vel")
    if at_data is not None:
        kw["at"] = at_data
    if at_vel_data is not None:
        kw["at_vel"] = at_vel_data
    data = cxfmapi.act(op, tau, x.data, x.chart, x.rep, **kw)
    return replace(x, data=data)


def _unwrap_anchor(anchor: Any, chart: Any, name: str, /) -> CDict | None:
    """Unwrap a Point/Tangent anchor to its data, checking the chart."""
    if anchor is None:
        return None
    if isinstance(anchor, (Point, Tangent)):
        if anchor.chart != chart:
            msg = (
                f"{name!r} chart {anchor.chart!r} does not match "
                f"the tangent's chart {chart!r}."
            )
            raise ValueError(msg)
        return cast("CDict", anchor.data)
    return cast("CDict", anchor)


@plum.dispatch
def act(op: cxfm.AbstractTransform, tau: Any, x: Point, /, **kw: Any) -> Point:
    """Act a frame transform on a Point.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cx.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")
    >>> vec = cx.Point.from_(q)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [km]
        [1 0 0]>

    >>> transformed_vec = cx.act(op, None, vec)
    >>> print(transformed_vec)
    <Point: chart=Cart3D (x, y, z) [km]
        [0 1 0]>

    """
    data = cxfmapi.act(op, tau, x.data, x.chart, x.rep, **kw)
    return replace(x, data=data)


@plum.dispatch
def act(
    op: cxfm.AbstractTransform, tau: Any, x: Coordinate, /, **kw: Any
) -> Coordinate:
    """Act a frame transform on a Coordinate (point + all fibres).

    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m", cxf.alice)
    >>> vel = cx.Tangent(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_basis, cxr.vel, frame=cxf.alice,
    ... )
    >>> pv = cx.Coordinate(point=point, velocity=vel)
    >>> pv_alex = pv.to_frame(cxf.alex)
    >>> pv_alex.frame
    Alex()
    >>> pv_alex["velocity"].frame
    Alex()

    A time-dependent transform prolongs the whole bundle jointly: a uniformly
    moving translation boosts the velocity fibre by its rate:

    >>> delta = lambda t: {"x": u.Q(3.0, "m/s") * t, "y": u.Q(0.0, "m"),
    ...                    "z": u.Q(0.0, "m")}
    >>> op = cx.Translate(delta, chart=cxc.cart3d)
    >>> out = cx.act(op, u.Q(2.0, "s"), pv)
    >>> out.point.data["x"], out["velocity"].data["x"]
    (Q(7., 'm'), Q(4., 'm / s'))

    """
    if _needs_joint_jet(op):
        # The jet anchors are structurally the bundle's own point and fibres;
        # caller-supplied anchor overrides are meaningless here — reject them
        # loudly rather than silently ignoring them (the static path below
        # honors an 'at' override, so silence would diverge invisibly).
        unsupported = set(kw) - {"usys"}
        if unsupported:
            msg = (
                "act on a Coordinate under a time-dependent transform "
                f"does not accept keyword overrides {sorted(unsupported)}: "
                "the bundle itself supplies the jet anchors."
            )
            raise TypeError(msg)
        return _act_coordinate_jet(op, tau, x, usys=kw.get("usys"))

    new_point = cxfm.act(op, tau, x.point, **kw)
    # Inject the base-point data as 'at' so non-Cartesian tangent dispatches
    # can evaluate the Jacobian at the correct location.  Callers may
    # override this by passing their own 'at' in **kw.
    # 'at' is resolved per-fibre: if the fibre's chart differs from the
    # point's chart (possible via _create_unchecked / cconvert with
    # field_charts=), we convert the base point into the fibre's chart first
    # so as not to violate Tangent's at-chart requirement.
    kw_base = dict(kw)
    new_fields: dict[str, Any] = {}
    for name, fibre in x._data.items():
        if "at" not in kw_base:
            fibre_kw = {**kw_base, "at": _point_data_in(x.point, fibre.chart)}
        else:
            fibre_kw = kw_base
        new_fields[name] = cxfm.act(op, tau, fibre, **fibre_kw)
    return Coordinate(point=new_point, **new_fields)


def _needs_joint_jet(op: cxfm.AbstractTransform, /) -> bool:
    """Whether a Coordinate bundle must be transformed as a joint jet.

    True when the op has time-dependent (callable) parameters, and also for
    `Boost`, whose *point action* is intrinsically tau-dependent (x + dv*tau)
    even when dv is a constant — its per-fibre closed forms then need the
    lower jet slots that only the joint path supplies.

    TODO: replace the isinstance test with a declared property on the
    transform (tracked with the TimeDep parameter refactor, issue #537).
    """
    if isinstance(op, cxfm.Boost):
        return True
    if isinstance(op, cxfm.Composed):
        return any(_needs_joint_jet(sub) for sub in op.transforms)
    return cxfm.is_time_dependent(op)


def _point_data_in(point: Point, chart: Any, /) -> CDict:
    """Return the point's data expressed in ``chart`` (no-op if it matches)."""
    if point.chart == chart:
        return point.data
    return cast("Point", cxr.cconvert(point, chart)).data


def _act_coordinate_jet(
    op: cxfm.AbstractTransform,
    tau: Any,
    x: Coordinate,
    /,
    *,
    usys: OptUSys = None,
) -> Coordinate:
    """Act a time-dependent transform on a Coordinate via joint prolongation.

    The point and all ladder fibres (velocity, acceleration, ...) form a jet
    of the underlying curve; the transform's kinematic prolongation is applied
    to the whole jet at once, so each fibre correctly gains the transform's
    time-derivative terms. Displacement fibres (same-tau point differences)
    transform by the frozen-tau pushforward at the base point.
    """
    point_chart = x.point.chart

    # Assemble the jet in the point's chart. Fibres in other charts are
    # converted in (and back out) via the Jacobian pushforward.
    jet: dict[int, CDict] = {0: x.point.data}
    ladder: dict[str, tuple[int, Any, Any]] = {}  # name -> (order, fibre, orig_chart)
    push_fibres: dict[str, Any] = {}
    for name, fibre in x._data.items():
        order = fibre.rep.semantic_kind.order
        if order is None or order == 0:
            push_fibres[name] = fibre
            continue
        orig_chart = fibre.chart
        f = fibre
        if orig_chart != point_chart:
            at_f = _point_data_in(x.point, orig_chart)
            f = cast("Tangent", cxrapi.cconvert(f, point_chart, at=at_f, usys=usys))
        if order in jet:
            msg = (
                f"Coordinate has multiple fibres at ladder order {order}; "
                "the joint prolongation under a time-dependent transform is "
                "ambiguous."
            )
            raise ValueError(msg)
        jet[order] = cast("CDict", f.data)
        ladder[name] = (order, f, orig_chart)

    out_jet = cast(
        "dict[int, CDict]", cxfmapi.prolong(op, tau, jet, point_chart, usys=usys)
    )
    new_point = replace(x.point, data=out_jet[0])

    new_fields: dict[str, Any] = {}
    for name, (order, f, orig_chart) in ladder.items():
        nf = replace(f, data=out_jet[order])
        if orig_chart != point_chart:
            nf = cast(
                "Tangent", cxrapi.cconvert(nf, orig_chart, at=out_jet[0], usys=usys)
            )
        new_fields[name] = nf

    # Displacement (and other non-ladder) fibres: frozen-tau pushforward
    # anchored at the pre-transform base point.
    for name, fibre in push_fibres.items():
        data = cxfmapi.pushforward(
            op,
            tau,
            fibre.data,
            fibre.chart,
            fibre.rep,
            at=_point_data_in(x.point, fibre.chart),
            usys=usys,
        )
        new_fields[name] = replace(fibre, data=data)

    return Coordinate(point=new_point, **new_fields)


@dataclassish.replace.dispatch  # type: ignore[attr-defined]
def _replace_coordinate(coord: Coordinate, /, **kwargs: Any) -> Coordinate:
    """Replace fields on a Coordinate.

    Supports replacing ``point``, any named fibre field, or ``frame``.  When
    ``frame`` is supplied, it is forwarded to both the base ``point`` and every
    fibre field so the bundle stays internally consistent.

    Unknown keys are rejected with a ``TypeError``.


    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m", cxf.alice)
    >>> pv = cx.Coordinate(point=point)
    >>> import dataclassish

    Replace the frame (propagates to point and all fibres):

    >>> pv2 = dataclassish.replace(pv, frame=cxf.alex)
    >>> pv2.frame
    Alex()

    Replace the base point directly:

    >>> point2 = cx.Point.from_([2.0, 0.0, 0.0], "m", cxf.alice)
    >>> pv3 = dataclassish.replace(pv, point=point2)
    >>> pv3.point is point2
    True

    """
    # Validate: reject unknown keys up front
    valid_keys = {"frame", "point"} | coord._data.keys()
    unknown = set(kwargs) - valid_keys
    if unknown:
        msg = (
            f"dataclassish.replace(Coordinate, ...): unknown field(s) {unknown!r}. "
            f"Valid fields are: {sorted(valid_keys)!r}."
        )
        raise TypeError(msg)

    frame = kwargs.pop("frame", None)
    new_point = kwargs.pop("point", coord.point)
    # Remaining kwargs are fibre-field replacements
    new_fields = dict(coord._data)
    new_fields.update(kwargs)

    if frame is not None:
        new_point = replace(new_point, frame=frame)
        new_fields = {name: replace(f, frame=frame) for name, f in new_fields.items()}

    return Coordinate(point=new_point, **new_fields)


# ===================================================================
# Coordinate conversion


@plum.dispatch
def cconvert(
    pv: Coordinate, to_chart: cxc.AbstractChart, /, *, usys: OptUSys = None
) -> Coordinate:
    """Convert a Coordinate to a new chart.

    Delegates to {meth}`Coordinate.cconvert`.

    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc

    >>> pt = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> pv = cx.Coordinate(point=pt)
    >>> pv_sph = cx.cconvert(pv, cxc.sph3d)
    >>> pv_sph.point.chart
    Spherical3D(M=Rn(3))

    """
    return pv.cconvert(to_chart, usys=usys)
