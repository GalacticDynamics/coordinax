"""Vector."""

__all__: tuple[str, ...] = ()


from typing import Any

import equinox as eqx
import plum

import coordinax._src.operators as cxo
import coordinax.charts as cxc
import coordinax.roles as cxr
import coordinax_api as cxapi
from .base import Vector
from coordinax._src import api
from coordinax._src.custom_types import CsDict

##############################################################################
# as_pos


@plum.dispatch
def as_pos(
    point: Vector,
    origin: Vector | None = None,
    /,
    *,
    chart: cxc.AbstractChart | None = None,  # type: ignore[type-arg]
    at: Vector | None = None,
    **kwargs: Any,
) -> Vector:
    r"""Convert a position vector to a displacement from the origin.

    Mathematical Definition:
    Given a position $p$ and an origin $o$, the displacement is:

    $$ \vec{d} = p - o \in T_o M $$

    Parameters
    ----------
    point
        Point vector to convert. Must have {class}`coordinax.roles.Point` role.
    origin
        Origin point. For embedded manifolds, this parameter is required.
    chart
        Target chart for the output displacement vector. If `None` (default),
        the displacement is returned in the chart computed from the input.
        If specified, the result is converted via {func}`~coordinax.vconvert`.
    at
        Base point for the chart conversion when ``chart`` is specified.
        Required for non-Cartesian target charts since tangent-space
        transformations depend on the position. Ignored if ``chart`` is `None`.
    **kwargs
        Additional keyword arguments passed to {func}`~coordinax.vconvert` when
        ``chart`` is specified.

    Returns
    -------
    Vector
        Displacement vector with ``PhysDisp`` role.

    Raises
    ------
    TypeError
        If ``pos`` does not have ``PhysDisp`` role, or if ``origin`` does not
        have ``PhysDisp`` role when provided.
    NotImplementedError
        For intrinsic manifolds without embedding, or for embedded manifolds
        when proper parallel transport is not yet implemented.

    Notes
    -----
    **Euclidean case:**
        Displacements are computed by converting both points to a shared
        Cartesian representation (using position transform), subtracting
        componentwise, then optionally converting via ``Pos.vconvert``
        to the requested chart.

    **Embedded manifold case:**
        Points are embedded to ambient Cartesian space, subtracted to get
        an ambient displacement, then optionally projected to the tangent
        space at ``at`` via ``Pos.vconvert``.

    **Intrinsic manifold case:**
        Not yet implemented. Raises ``NotImplementedError``.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Convert a point to a displacement (from the coordinate origin):

    >>> point = cx.Vector.from_([1, 2, 3], "m")
    >>> print(point)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> disp = cx.as_pos(point)
    >>> print(disp)
    <Vector: chart=Cart3D, role=PhysDisp (x, y, z) [m]
        [1 2 3]>

    Convert with an explicit origin:

    >>> origin = cx.Vector.from_([0.5, 0.5, 0.5], "m")
    >>> disp = cx.as_pos(point, origin)
    >>> print(disp)
    <Vector: chart=Cart3D, role=PhysDisp (x, y, z) [m]
        [0.5 1.5 2.5]>

    Request a specific representation (uses PhysDisp.vconvert):

    >>> pos_sph = point.vconvert(cxc.sph3d)
    >>> disp_sph = cx.as_pos(pos_sph, origin, chart=cxc.cyl3d, at=pos_sph)
    >>> disp_sph.chart
    Cylindrical3D()

    """
    point = eqx.error_if(
        point, not isinstance(point.role, cxr.Point), "point is not a point vector."
    )
    if origin is None:
        disp_data, disp_chart = api.as_pos(point.data, point.chart, origin)

    else:
        origin = eqx.error_if(
            origin,
            not isinstance(origin.role, cxr.Point),
            "origin is not a point vector.",
        )
        # Compute displacement in a shared chart (for embedded, in ambient space)
        origin = origin.vconvert(point.chart)
        disp_data, disp_chart = api.as_pos(point.data, point.chart, origin.data)

    # Create displacement vector in computed chart
    disp_vec = Vector(disp_data, disp_chart, cxr.phys_disp)
    if chart is not None:
        # Pass `at` as positional argument to match dispatch signature:
        # vconvert(role, to_chart, from_vec, from_pos)
        disp_vec = disp_vec.vconvert(chart, at, **kwargs)
    return disp_vec


##############################################################################
# Vector conversion


@plum.dispatch
def vconvert(
    role: cxr.Point,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_vec: Vector,
    /,
) -> Vector:
    """Convert a vector from one representation to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.vconvert(cxc.sph3d, vec)
    >>> print(sph_vec)
    <Vector: chart=Spherical3D, role=Point (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    from_vec = eqx.error_if(
        from_vec,
        not isinstance(from_vec.role, cxr.Point),
        "from_vec is not a point vector and requires a point vector "
        "for the change-of-basis.",
    )
    # Call the `vconvert` function on the data from the vector's kind
    p = cxapi.vconvert(from_vec.role, to_chart, from_vec.chart, from_vec.data)
    # Return a new vector
    return Vector(data=p, chart=to_chart, role=role)


@plum.dispatch
def vconvert(
    role: cxr.PhysDisp,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_vec: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    r"""Convert a position-difference (physical displacement) vector.

    From one representation to another.

    Mathematical Definition:

    A **position difference** (PhysDisp role) is a tangent vector v \in T_p M.
    It transforms via the pushforward (Jacobian) at the base point p:

    $$ v_S = J_{R \to S}(p) \, v_R $$
    This is the same transformation rule as velocity and acceleration, but
    PhysDisp has units of length (not length/time).

    Parameters
    ----------
    role
        The position-difference role flag.
    to_chart : AbstractChart
        Target representation.
    from_vec : Vector
        Position-difference vector to transform.
    from_pos : Vector
        Base point (position) at which to evaluate the transformation.
        Must have `Point` role.

    Returns
    -------
    Vector
        Transformed position-difference in the target representation.

    Notes
    -----
    - Pos transforms via **physical_tangent_transform** (tangent space transformation).
    - Point transforms via **point_transform** (coordinate transformation).
    - This distinction is fundamental in differential geometry.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> # A position-difference and a base point
    >>> disp = cx.Vector(
    ...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    ...     cxc.cart3d,
    ...     cxr.phys_disp,
    ... )
    >>> point = cx.Vector.from_([1, 1, 1], "m")

    >>> # Transform to spherical - requires base point
    >>> sph_disp = cx.vconvert(cxc.sph3d, disp, point)
    >>> sph_disp.role
    PhysDisp()

    """
    from_vec = eqx.error_if(
        from_vec,
        not isinstance(from_vec.role, cxr.PhysDisp),
        "from_vec is not a position-difference vector.",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.role, cxr.Point),
        "'from_pos' must be a point vector",
    )

    # Convert the base point to the displacement's chart
    from_pos = from_pos.vconvert(from_vec.chart)

    # Pos transforms via physical_tangent_transform (tangent space / pushforward)
    # This is the SAME rule as Vel/Acc, just different units
    p = cxapi.vconvert(
        from_vec.role, to_chart, from_vec.chart, from_vec.data, from_pos.data
    )
    # Return a new vector with Pos role preserved
    return Vector(data=p, chart=to_chart, role=role)


@plum.dispatch
def vconvert(to_chart: cxc.AbstractChart, from_vec: Vector, /) -> Vector:  # type: ignore[type-arg]
    """Convert a vector from one representation to another."""
    # Redispatch to the role-specific version
    return cxapi.vconvert(from_vec.role, to_chart, from_vec)


@plum.dispatch
def vconvert(
    role: cxr.PhysVel | cxr.PhysAcc,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_dif: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    """Convert a vector from one differential to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> qvec = cx.Vector.from_([1, 1, 1], "m")
    >>> vvec = cx.Vector.from_([10, 10, 10], "m/s")

    >>> sph_vvec = cx.vconvert(cxc.sph3d, vvec, qvec)

    """
    # Checks
    from_dif = eqx.error_if(
        from_dif,
        isinstance(from_dif.role, cxr.Point),
        "'from_dif' must be a differential vector",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.role, cxr.Point),
        "'from_pos' must be a point vector",
    )

    # Convert the base point to the differential's chart
    from_pos = from_pos.vconvert(from_dif.chart)

    # Call the `vconvert` function on the data from the vector's kind
    p = cxapi.vconvert(
        from_dif.role, to_chart, from_dif.chart, from_dif.data, from_pos.data
    )
    # Return a new vector
    return Vector(data=p, chart=to_chart, role=from_dif.role)


@plum.dispatch
def vconvert(
    role: cxr.CoordDisp | cxr.CoordVel | cxr.CoordAcc,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_dif: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    """Convert a coordinate-basis tangent vector between charts (requires at=)."""
    from_dif = eqx.error_if(
        from_dif,
        isinstance(from_dif.role, cxr.Point),
        "'from_dif' must be a differential vector",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.role, cxr.Point),
        "'from_pos' must be a point vector",
    )

    from_pos = from_pos.vconvert(from_dif.chart)
    p = cxapi.vconvert(
        from_dif.role, to_chart, from_dif.chart, from_dif.data, from_pos.data
    )
    return Vector(data=p, chart=to_chart, role=from_dif.role)


@plum.dispatch
def vconvert(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_dif: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    """Convert a vector from one representation to another."""
    # Redispatch to the role-specific version
    return cxapi.vconvert(from_dif.role, to_chart, from_dif, from_pos)


# =============================================================================
# Operator dispatch


@plum.dispatch
def apply_op(
    op: cxo.AbstractOperator,
    tau: Any,
    v: Vector,
    /,
    *,
    at: Vector | None = None,
    **kw: Any,
) -> Vector:
    """Apply an operator to a Vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.ops as cxo

    >>> v = cx.Vector.from_({"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")})
    >>> op = cxo.Rotate.from_euler("z", u.Q(90, "deg"))

    >>> v_rot = cxo.apply_op(op, None, v)  # no time dependence
    >>> print(v_rot)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [-2.  1.  3.]>

    """
    ats: dict[str, Any] = {}
    other_kw: dict[str, Any] = {}

    if at is not None:
        at_vec = at if at.chart == v.chart else at.vconvert(v.chart)
        ats["at"] = at_vec.data

    for k, val in kw.items():
        if not k.startswith("at_"):
            other_kw[k] = val
            continue

        if val is None:
            continue
        if isinstance(val, Vector):
            ats[k] = val.vconvert(v.chart).data
        elif isinstance(val, dict):
            ats[k] = val
        else:
            msg = f"{k} must be a Vector, CsDict, or None"
            raise TypeError(msg)

    # Apply to the underlying data with the vector's role & chart.
    result_data = api.apply_op(op, tau, v.role, v.chart, v.data, **(other_kw | ats))

    # Return a new Vector with the same chart and role
    return Vector(data=result_data, chart=v.chart, role=v.role)


@plum.dispatch
def apply_op(
    op: cxo.Identity,
    tau: Any,
    v: Vector,
    /,
    *,
    at: Vector | None = None,
    **kw: Any,
) -> Vector:
    """Apply Identity operator to a Vector - returns input unchanged.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.ops as cxo
    >>> import unxt as u

    >>> v = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> op = cxo.Identity()
    >>> op(v) is v
    True

    """
    del op, tau, at, kw  # unused
    return v


# ===================================================================
# cdict dispatch


@plum.dispatch
def cdict(obj: Vector, /) -> CsDict:
    """Extract component dictionary from a Vector.

    Parameters
    ----------
    obj
        A Vector object

    Returns
    -------
    dict[str, Any]
        The component dictionary from the vector's data field.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    return obj.data
