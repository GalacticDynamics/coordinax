"""Point-roled transformations in the same atlas."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable
from jaxtyping import Array
from typing import Any, Final, cast

import jax
import plum

import quaxed.numpy as jnp
import unxt as u
import unxts.linalg as ul
from unxt import AbstractQuantity as ABCQ  # noqa: N814

import coordinaxs.api.charts as cxcapi
from .checks import check_manifolds_match_charts
from .containers import canonical_containers
from .d0 import Cart0D
from .d1 import Cart1D, Radial1D, Time1D
from .d2 import Cart2D, Polar2D
from .d3 import (
    AbstractSpherical3D,
    Cart3D,
    Cylindrical3D,
    LonCosLatSpherical3D,
    LonLatSpherical3D,
    MathSpherical3D,
    ProlateSpheroidal3D,
    Spherical3D,
)
from .d6 import PoincarePolar6D
from .dn import CartND
from coordinax._src.base import AbstractChart
from coordinax._src.base.manifold import AbstractManifold
from coordinax._src.custom_types import OptUSys
from coordinax._src.euclidean import RN, EuclideanManifold, Rn
from coordinax._src.exceptions import ManifoldMismatchError
from coordinax._src.null import NoManifold
from coordinax._src.product.chart import CartesianProductChart
from coordinax._src.product.manifold import CartesianProductManifold
from coordinax._src.utils import complement_angle, rad_value, strip, wrap, wrap_angle
from coordinaxs.api.custom_types import CDict


def _ratio_zero_on_axis(num: Array, denom: Array, /) -> Array:
    """``num / denom``, defined as 0 where ``denom == 0`` (coordinate singularity).

    Uses the double-``where`` idiom so that both the value *and* its gradient are
    finite on the axis. A plain ``jnp.where(denom == 0, 0, num / denom)`` still
    evaluates ``num / denom`` in the unselected branch, leaking ``NaN`` into
    reverse-mode gradients (``0 * inf``); guarding the denominator first avoids it.
    """
    safe = jnp.where(denom == 0, jnp.ones_like(denom), denom)
    ratio = num / safe
    return jnp.where(denom == 0, jnp.zeros_like(ratio), ratio)


_PROLATE_NEEDS_USYS: Final = (
    "For non-Quantity 'mu' or 'nu', usys must be a UnitSystem, not None."
)


def _delta_squared(chart: Any, unit_area: Any, usys: OptUSys, /) -> Any:
    """``Delta**2`` as a raw value in *unit_area*, or in ``usys`` when bare.

    A bare ``mu``/``nu`` carries no unit to measure ``Delta`` against, so the
    unit system is required there -- which is also what narrows `usys` to
    non-`None` before it is subscripted.
    """
    if unit_area is not None:
        # Strip `Delta` into the implied *length* unit and square the raw
        # array, rather than squaring the `Quantity` and stripping the area:
        # `Quantity ** 2` is one of the per-primitive `quax` traces this is
        # here to avoid, and both orders give the same number.
        return cast("Array", u.ustrip(unit_area**0.5, chart.Delta)) ** 2
    if usys is None:
        raise ValueError(_PROLATE_NEEDS_USYS)
    return cast("Array", u.ustrip(usys["length"], chart.Delta)) ** 2


def _require_cart3d_phase_space(chart: Any, /, *, direction: str) -> None:
    """Validate a two-factor ``Cart3D`` Cartesian phase-space product chart.

    ``direction`` is ``"to"`` (``chart`` is the source of ``pt_map`` *to*
    ``PoincarePolar6D``) or ``"from"`` (``chart`` is the target of ``pt_map``
    *from* it); it only selects the error wording. Raises ``NotImplementedError``
    unless ``chart`` has exactly two ``Cart3D`` factors [position, velocity].
    """
    if len(chart.factors) != 2 or not all(isinstance(f, Cart3D) for f in chart.factors):
        role = "source" if direction == "to" else "target"
        msg = (
            f"pt_map {direction} PoincarePolar6D requires a Cartesian phase-space "
            f"{role}: a two-factor CartesianProductChart of (Cart3D, Cart3D) "
            f"[position, velocity]; got factors {chart.factors!r}."
        )
        raise NotImplementedError(msg)


#####################################################################
# Point transformations

# ===================================================================
# Partial application


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def pt_map(q: None, /, *fixed_args: Any, **fixed_kw: Any) -> Callable[..., Any]:
    """Return a partial function for point transformation.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Coordinates without units are the default.

    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d)
    >>> map(q)
    {'r': Q(1., 'm'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    Coordinates without units are also accepted, interpreted having units of the
    `unxt.AbstractUnitSystem`, which must be passed.

    >>> q = {"x": 1.0, "y": 0.0, "z": 0.0}
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(q)
    {'r': Array(1., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64, ...),
     'phi': Array(0., dtype=float64, ...)}

    `unxt.Quantity` inputs are also accepted, and are interpreted as being in
    Cartesian coordinates.

    >>> p = u.Q([1.0, 0.0, 0.0], "m")
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d)
    >>> map(p)
    QM([1.        , 1.57079633, 0.        ], '(m, rad, rad)')

    Array-Like inputs are interpreted as Cartesian coordinates with units from
    the required `unxt.AbstractUnitSystem`.

    >>> q = [1.0, 0.0, 0.0]
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(q)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    """
    del q  # unused

    # NOTE: lambda is much faster than ft.partial here
    return lambda x, *args, **kw: cxcapi.pt_map(x, *fixed_args, *args, **fixed_kw, **kw)


@plum.dispatch
def pt_map(
    from_chart: AbstractChart, to_chart: AbstractChart, /, **fixed_kw: Any
) -> Callable[..., Any]:
    """Return a partial function for point transformation.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Coordinates without units are the default.

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d)
    >>> map(p)
    {'r': Q(1., 'm'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    Coordinates without units are also accepted, interpreted having units of the
    `unxt.AbstractUnitSystem`, which must be passed.

    >>> p = {"x": 1.0, "y": 0.0, "z": 0.0}
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(p)
    {'r': Array(1., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64, ...),
     'phi': Array(0., dtype=float64, ...)}

    `unxt.Quantity` inputs are also accepted, and are interpreted as being in
    Cartesian coordinates.

    >>> p = u.Q([1.0, 0.0, 0.0], "m")
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d)
    >>> map(p)
    QM([1.        , 1.57079633, 0.        ], '(m, rad, rad)')

    Array-Like inputs are interpreted as Cartesian coordinates with units from
    the required `unxt.AbstractUnitSystem`.

    >>> p = [1.0, 0.0, 0.0]
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(p)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    """
    out = cxcapi.pt_map(None, from_chart, to_chart, **fixed_kw)
    return cast("Callable[..., Any]", out)


# ===================================================================
# Redispatch with the manifold


@plum.dispatch
def pt_map(
    x: Any,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Point transformation from chart to chart, using their manifolds.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {}
    >>> cxc.pt_map(p, cxc.cart0d, cxc.cart0d)
    {}

    >>> p = {"r": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxc.radial1d, cxc.cart1d)
    {'x': Q(5., 'm')}

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxc.polar2d, cxc.cart2d)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxc.sph3d, cxc.cart3d)
    {'x': Q(5., 'm'), 'y': Q(0., 'm'), 'z': Q(3.061617e-16, 'm')}

    """
    return cxcapi.pt_map(x, from_chart.M, from_chart, to_chart.M, to_chart, usys=usys)


# ===================================================================
# General representation conversions


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def pt_map(
    p: CDict,
    from_M: AbstractManifold,
    from_chart: AbstractChart,
    to_M: AbstractManifold,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> Cartesian -> AbstractChart.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> map = cxc.pt_map.invoke(dict[str, u.Q], cxm.Rn, cxc.AbstractChart,
    ...                         cxm.Rn, cxc.AbstractChart)
    >>> map(p, cxm.R2, cxc.polar2d, cxm.R2, cxc.cart2d)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    # Even though there's a dispatch for the Self-to-Self case, we still check
    # for it here to avoid infinite recursion.
    if from_chart == to_chart:
        return canonical_containers(p, to_chart)

    # Now we know from_chart and to_chart are different, so we can safely call.
    from_cart = from_chart.cartesian
    to_cart = to_chart.cartesian
    if from_cart != to_cart:
        msg = (
            f"no transition from {from_chart} to {to_chart}: their Cartesian "
            f"charts differ ({from_cart} vs {to_cart})"
        )
        raise ManifoldMismatchError(msg)

    p_cart = cxcapi.pt_map(p, from_M, from_chart, to_M, from_cart, usys=usys)
    p_out = cxcapi.pt_map(p_cart, from_M, from_cart, to_M, to_chart, usys=usys)
    return canonical_containers(cast("CDict", p_out), to_chart)


##############################################################################
# Transition Maps Assuming on Same Atlas

# ===================================================================
# Self representation conversions

IDENTITY_TRANSFORM_CHARTS: Final[tuple[type[AbstractChart[Any, Any, Any]], ...]] = (
    # 0D
    Cart0D,
    # 1D
    Cart1D,
    Radial1D,
    Time1D,
    # 2D
    Cart2D,
    Polar2D,
    # 3D
    Cart3D,
    Cylindrical3D,
    Spherical3D,
    LonLatSpherical3D,
    LonCosLatSpherical3D,
    MathSpherical3D,
    # ProlateSpheroidal3D,  # requires Delta
    # 6D
    PoincarePolar6D,
    # N-D
    CartND,
    # MinkowskiCT is registered separately via MinkowskiAtlas
)


@plum.dispatch.multi(*((CDict, Rn, typ, Rn, typ) for typ in IDENTITY_TRANSFORM_CHARTS))
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Identity conversion for matching charts.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> q = {}
    >>> q2 = cxc.pt_map(q, cxm.R0, cxc.cart0d, cxm.R0, cxc.cart0d)
    >>> q is q2
    True

    >>> q = {"r": u.Q(3.0, "m")}
    >>> q2 = cxc.pt_map(q, cxm.R1, cxc.radial1d, cxm.R1, cxc.radial1d)
    >>> q is q2
    True

    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> q2 = cxc.pt_map(q, cxm.R2, cxc.cart2d, cxm.R2, cxc.cart2d)
    >>> q is q2
    True

    """
    del usys  # unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    return canonical_containers(p, to_chart)


# ---------------------------------------------------------
# Specific representation conversions

# -----------------------------------------------
# 1D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Radial1D,
    to_M: Rn,
    to_chart: Cart1D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Radial1D -> Cart1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> q = {"r": u.Q(5.0, "m")}
    >>> cxc.pt_map(q, cxm.R1, cxc.radial1d, cxm.R1, cxc.cart1d)
    {'x': Q(5., 'm')}

    >>> q = {"r": 5.0}  # No units
    >>> cxc.pt_map(q, cxm.R1, cxc.radial1d, cxm.R1, cxc.cart1d)
    {'x': 5.0}

    """
    del usys  # unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    return canonical_containers({"x": p["r"]}, to_chart)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart1D,
    to_M: Rn,
    to_chart: Radial1D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart1D -> Radial1D.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.

    Assumptions:

    - Cart1D and Radial1D are

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxm.R1, cxc.cart1d, cxm.R1, cxc.radial1d)
    {'r': Q(5., 'm')}

    >>> p = {"x": 5.0}  # No units
    >>> cxc.pt_map(p, cxm.R1, cxc.cart1d, cxm.R1, cxc.radial1d)
    {'r': 5.0}

    """
    del usys  # unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    return canonical_containers({"r": p["x"]}, to_chart)


# -----------------------------------------------
# 2D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Polar2D,
    to_M: Rn,
    to_chart: Cart2D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Polar2D -> Cart2D.

    The `r` and `theta` coordinates are converted to the `x` and `y` coordinates
    of the 2D Cartesian system.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxm.R2, cxc.polar2d, cxm.R2, cxc.cart2d)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    >>> p = {"r": 5, "theta": 90}  # No units
    >>> usys = u.unitsystem("km", "deg")
    >>> cxc.pt_map(p, cxm.R2, cxc.polar2d, cxm.R2, cxc.cart2d, usys=usys)
    {'x': Array(3.061617e-16, dtype=float64, ...),
     'y': Array(5., dtype=float64, ...)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    # `strip`/`rad_value`/`wrap` keep the arithmetic off `Quantity` operands,
    # where every primitive costs a `quax` trace; see `coordinax._src.utils`.
    (r_,), unit = strip(p, ("r",))
    theta = rad_value(p["theta"], usys)
    return canonical_containers(
        {"x": wrap(r_ * jnp.cos(theta), unit), "y": wrap(r_ * jnp.sin(theta), unit)},
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart2D,
    to_M: Rn,
    to_chart: Polar2D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart2D -> Polar2D.

    The `x` and `y` coordinates are converted to the `r` and `theta` coordinates
    of the 2D polar system.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(3, "m"), "y": u.Q(4, "m")}
    >>> cxc.pt_map(p, cxm.R2, cxc.cart2d, cxm.R2, cxc.polar2d)
    {'r': Q(5., 'm'), 'theta': Angle(0.92729522, 'rad')}

    >>> p = {"x": 3, "y": 4}  # No units
    >>> cxc.pt_map(p, cxm.R2, cxc.cart2d, cxm.R2, cxc.polar2d)
    {'r': Array(5., dtype=float64, ...),
     'theta': Array(0.92729522, dtype=float64, ...)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    (x, y), unit = strip(p, ("x", "y"))
    return canonical_containers(
        {
            "r": wrap(jnp.hypot(x, y), unit),
            "theta": wrap_angle(jnp.arctan2(y, x), unit, usys),
        },
        to_chart,
    )


# -----------------------------------------------
# 3D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cylindrical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cylindrical3D -> Cart3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"rho": u.Q(1.0, "m"), "phi": u.Q(90, "deg"), "z": u.Q(2.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.cart3d)
    {'x': Q(6.123234e-17, 'm'), 'y': Q(1., 'm'), 'z': Q(2., 'm')}

    >>> p = {"rho": 1.0, "phi": 90, "z": 2.0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.cart3d, usys=usys)
    {'x': Array(6.123234e-17, dtype=float64, ...), 'y': Array(1., dtype=float64, ...),
     'z': 2.0}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    # `z` is not consumed by the arithmetic, so it is carried through untouched
    # and keeps its own unit and container.
    (rho,), unit = strip(p, ("rho",))
    phi = rad_value(p["phi"], usys)
    return canonical_containers(
        {
            "x": wrap(rho * jnp.cos(phi), unit),
            "y": wrap(rho * jnp.sin(phi), unit),
            "z": p["z"],
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> Cart3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(0., 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the equator (theta=90 deg, phi=0):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cart3d, usys=usys)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    (r_,), unit = strip(p, ("r",))
    theta = rad_value(p["theta"], usys)
    phi = rad_value(p["phi"], usys)
    # `sin(theta)` once rather than once per component.
    rho = r_ * jnp.sin(theta)
    return canonical_containers(
        {
            "x": wrap(rho * jnp.cos(phi), unit),
            "y": wrap(rho * jnp.sin(phi), unit),
            "z": wrap(r_ * jnp.cos(theta), unit),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: LonLatSpherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """LonLatSpherical3D -> Cart3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point at the north pole (lat=90 deg):

    >>> p = {"lon": u.Q(0, "deg"), "lat": u.Q(90, "deg"), "distance": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.lonlat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(6.123234e-17, 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the equator at lon=0:

    >>> p = {"lon": 0, "lat": 0, "distance": 2}
    >>> cxc.pt_map(p, cxm.R3, cxc.lonlat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(0., dtype=float64, ...)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    (r_,), unit = strip(p, ("distance",))
    lon = rad_value(p["lon"], usys)
    lat = rad_value(p["lat"], usys)
    # `cos(lat)` once rather than once per component.
    rho = r_ * jnp.cos(lat)
    return canonical_containers(
        {
            "x": wrap(rho * jnp.cos(lon), unit),
            "y": wrap(rho * jnp.sin(lon), unit),
            "z": wrap(r_ * jnp.sin(lat), unit),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: LonCosLatSpherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """LonCosLatSpherical3D -> Cart3D.

    Components are (lon_coslat, lat, r), where lon_coslat := lon * cos(lat).
    Longitude is undefined at the poles (cos(lat) == 0); we set lon = 0 by
    convention there to avoid NaNs.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the equator (lat=0, so lon_coslat = lon):

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
    ...      "distance": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.loncoslat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(1., 'm'), 'y': Q(0., 'm'), 'z': Q(0., 'm')}

    At the north pole (lat=90), lon_coslat is effectively 0 regardless of lon:

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(90, "deg"),
    ...      "distance": u.Q(2.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.loncoslat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(1.2246468e-16, 'm'), 'y': Q(0., 'm'), 'z': Q(2., 'm')}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    (r_,), unit = strip(p, ("distance",))
    lat = rad_value(p["lat"], usys)
    # To radians *before* the division rather than after: `coslat` is
    # dimensionless, so the two commute, and dividing raw values keeps the
    # arithmetic off `Quantity` operands.
    lon_coslat = rad_value(p["lon_coslat"], usys)
    # Longitude is undefined at the poles. The guard fires only when cos(lat) is
    # *exactly* 0 (giving lon = 0); a floating-point near-pole value (e.g.
    # cos(pi/2) ~= 6e-17) still divides, but the cos(lat) factor below multiplies
    # the result back so x, y stay finite.
    coslat = jnp.cos(lat)
    lon = _ratio_zero_on_axis(lon_coslat, coslat)
    # Convert to Cartesian. `cos(lat)` is already in hand as `coslat`.
    rho = r_ * coslat
    return canonical_containers(
        {
            "x": wrap(rho * jnp.cos(lon), unit),
            "y": wrap(rho * jnp.sin(lon), unit),
            "z": wrap(r_ * jnp.sin(lat), unit),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: MathSpherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """MathSpherical3D -> Cart3D.

    - theta: azimuth in the x-y plane (longitude-like)
    - phi  : polar angle from +z, with phi in [0, pi]

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis (phi=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(0., 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the +x axis (theta=0, phi=90):

    >>> p = {"r": u.Q(2.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(2., 'm'), 'y': Q(0., 'm'), 'z': Q(1.2246468e-16, 'm')}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    (r_,), unit = strip(p, ("r",))
    theta = rad_value(p["theta"], usys)
    phi = rad_value(p["phi"], usys)
    # `sin(phi)` once rather than once per component.
    rho = r_ * jnp.sin(phi)
    return canonical_containers(
        {
            "x": wrap(rho * jnp.cos(theta), unit),
            "y": wrap(rho * jnp.sin(theta), unit),
            "z": wrap(r_ * jnp.cos(phi), unit),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: ProlateSpheroidal3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""ProlateSpheroidal3D -> Cart3D.

    We calculate through cylindrical coordinates first:

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{\lvert\nu\rvert}{\Delta^2}\right)}$
    $z = \sqrt{\mu\,\frac{\lvert\nu\rvert}{\Delta^2}}\;\mathrm{sign}(\nu)$
    $\phi = \phi.$

    Then convert to Cartesian:

    $x=\rho\cos\phi$, $y=\rho\sin\phi$, $z=z$.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cart3d)
    {'x': Q(0.8660254, 'm'), 'y': Q(0., 'm'), 'z': Q(1.11803399, 'm')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cart3d, usys=usys)
    {'x': Array(0.8660254, dtype=float64),
     'y': Array(0., dtype=float64),
     'z': Array(1.11803399, dtype=float64)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    # Calculate cylindrical distance
    (mu, nu), unit_area = strip(p, ("mu", "nu"))
    delta2 = _delta_squared(from_chart, unit_area, usys)

    nu_d2 = jnp.abs(nu) / delta2
    rho = jnp.sqrt((mu - delta2) * (1 - nu_d2))

    # Convert to Cartesian. `mu` and `nu` are *areas*, so a length output
    # carries the square root of their unit.
    unit_len = None if unit_area is None else unit_area**0.5
    phi = rad_value(p["phi"], usys)
    return canonical_containers(
        {
            "x": wrap(rho * jnp.cos(phi), unit_len),
            "y": wrap(rho * jnp.sin(phi), unit_len),
            "z": wrap(jnp.sqrt(mu * nu_d2) * jnp.sign(nu), unit_len),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D,
    to_M: Rn,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Cylindrical3D.

    >>> import coordinax as cx
    >>> import unxt as u

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.cyl3d)
    {'rho': Q(5., 'm'), 'phi': Angle(0.92729522, 'rad'), 'z': Q(5., 'm')}

    >>> p = {"x": 3.0, "y": 4.0, "z": 5.0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.cyl3d)
    {'rho': Array(5., dtype=float64, ...),
     'phi': Array(0.92729522, dtype=float64, ...),
     'z': 5.0}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    # Only the components the arithmetic consumes are stripped; `z` is carried
    # through untouched so it keeps its own unit and container, exactly as the
    # `Quantity`-operand form did.
    (x, y), unit = strip(p, ("x", "y"))
    return canonical_containers(
        {
            "rho": wrap(jnp.hypot(x, y), unit),
            "phi": wrap_angle(jnp.atan2(y, x), unit, usys),
            "z": p["z"],
        },
        to_chart,
    )


@plum.dispatch.multi(
    (CDict, EuclideanManifold, Cart3D, EuclideanManifold, AbstractSpherical3D),
    (CDict, EuclideanManifold, Cylindrical3D, EuclideanManifold, AbstractSpherical3D),
)
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D | Cylindrical3D,
    to_M: Rn,
    to_chart: AbstractSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Spherical3D -> AbstractSpherical3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.loncoslat_sph3d)
    {'lon_coslat': Angle(0., 'rad'), 'lat': Angle(90., 'deg'), 'distance': Q(1., 'm')}

    >>> p = {"rho": 0, "phi": 180, "z": 1}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.loncoslat_sph3d, usys=usys)
    {'lon_coslat': Array(1.10218212e-14, dtype=float64, ...),
     'lat': Array(90., dtype=float64, ...),
     'distance': Array(1., dtype=float64, weak_type=True)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    # from_chart -> Spherical3D -> to_chart
    sph3d = Spherical3D(M=from_chart.M)
    p_sph = cxcapi.pt_map(p, from_M, from_chart, to_M, sph3d, usys=usys)
    out = cxcapi.pt_map(p_sph, from_M, sph3d, to_M, to_chart, usys=usys)
    return canonical_containers(cast("CDict", out), to_chart)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D,
    to_M: Rn,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Spherical3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis:

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Angle(0., 'rad'), 'phi': Angle(0., 'rad')}

    A point on the +x axis:

    >>> p = {"x": 2.0, "y": 0.0, "z": 0.0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.sph3d)
    {'r': Array(2., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64, ...),
     'phi': Array(0., dtype=float64, ...)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    # `strip`/`wrap` keep the arithmetic off `Quantity` operands, where every
    # primitive costs a `quax` trace; see the note in `coordinax._src.utils`.
    (x, y, z), unit = strip(p, ("x", "y", "z"))
    # `hypot`/`atan2` rather than `sqrt(x**2 + y**2 + z**2)` and `acos(z / r)`:
    # squaring over/underflows a decade and a half short of the float range
    # (`r` of a 10 kpc position in metres is `inf` in float32), and `acos`
    # saturates as `z / r -> 1`, losing every digit of `theta` near the poles.
    # Both `atan2` calls handle their own singular point, so no `where` guard
    # is needed: `atan2(0, 0) == 0` keeps the r == 0 convention theta == 0, and
    # phi == 0 on the z axis. This matches `Cart2D -> Polar2D`, which is
    # already written this way.
    rho = jnp.hypot(x, y)
    return canonical_containers(
        {
            "r": wrap(jnp.hypot(rho, z), unit),
            "theta": wrap_angle(jnp.atan2(rho, z), unit, usys),
            "phi": wrap_angle(jnp.atan2(y, x), unit, usys),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cylindrical3D,
    to_M: Rn,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cylindrical3D -> Spherical3D.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Angle(0., 'rad'), 'phi': Angle(0, 'rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": 3.0, "phi": 0, "z": 0.0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.sph3d)
    {'r': Array(3., dtype=float64, ...), 'theta': Array(1.57079633, dtype=float64, ...),
     'phi': 0}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    # `phi` carries an angle, so it is not part of this length-dimensioned
    # group and passes through untouched.
    (rho, z), unit = strip(p, ("rho", "z"))
    # `atan2(rho, z)` rather than `acos(z / r)`, which saturates as `z / r -> 1`
    # and loses every digit of `theta` near the poles. `atan2(0, 0) == 0` keeps
    # the r == 0 convention theta == 0, so the `where` guard is not needed.
    #
    # `abs(rho)` because `atan2` is sign-sensitive in its first argument where
    # `acos(z / hypot(rho, z))` was not: `hypot` squares `rho` away, so a
    # negative `rho` used to give the same `theta` as its positive twin.
    # `Cylindrical3D` does not value-validate `rho`, so a hand-built negative
    # one is reachable, and without this it would produce a negative `theta`
    # outside the `[0, pi]` that `Spherical3D.check_data` enforces.
    # `Cart3D -> Spherical3D` needs no such guard: its `rho` is a `hypot`.
    return canonical_containers(
        {
            "r": wrap(jnp.hypot(rho, z), unit),
            "theta": wrap_angle(jnp.atan2(jnp.abs(rho), z), unit, usys),
            "phi": p["phi"],
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> Cylindrical3D.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cyl3d)
    {'rho': Q(0., 'm'), 'phi': Angle(0, 'rad'), 'z': Q(1., 'm')}

    A point on the equator (theta=90 deg):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cyl3d, usys=usys)
    {'rho': Array(2., dtype=float64, ...), 'phi': 0,
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    # `phi` is an angle, outside this length group, and passes through untouched.
    (r_,), unit = strip(p, ("r",))
    theta = rad_value(p["theta"], usys)
    return canonical_containers(
        {
            "rho": wrap(r_ * jnp.sin(theta), unit),
            "phi": p["phi"],
            "z": wrap(r_ * jnp.cos(theta), unit),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: LonLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> LonLatSpherical3D.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Spherical theta=0 corresponds to lat=90 (north pole):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.lonlat_sph3d)
    {'lon': Angle(0, 'rad'), 'lat': Angle(90., 'deg'), 'distance': Q(1., 'm')}

    Spherical theta=90 deg corresponds to lat=0 (equator):

    >>> p = {"r": 1.0, "theta": 0, "phi": 0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.lonlat_sph3d)
    {'lon': 0, 'lat': 1.5707963267948966, 'distance': 1.0}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    lat = complement_angle(p["theta"], usys)
    return canonical_containers(
        {"lon": p["phi"], "lat": lat, "distance": p["r"]}, to_chart
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: LonCosLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> LonCosLatSpherical3D.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    On the equator (theta=90 deg), lon_coslat equals lon:

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.loncoslat_sph3d)
    {'lon_coslat': Angle(45., 'deg'), 'lat': Angle(0, 'deg'), 'distance': Q(1., 'm')}

    At the north pole (theta=0), lon_coslat = 0 regardless of phi:

    >>> p = {"r": 1.0, "theta": 0, "phi": 45}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.loncoslat_sph3d, usys=usys)
    {'lon_coslat': Array(2.7554553e-15, dtype=float64, ...),
     'lat': 90.0, 'distance': 1.0}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    lat = complement_angle(p["theta"], usys)
    # `cos` needs radians, and `lat` is in whatever unit the output carries --
    # `usys["angle"]` on the bare path, which need not be radians.
    lat_rad = jnp.pi / 2 - rad_value(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat_rad)
    return canonical_containers(
        {"lon_coslat": lon_coslat, "lat": lat, "distance": p["r"]}, to_chart
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: MathSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> MathSpherical3D.

    Swaps theta and phi: Physics (theta=polar, phi=azimuth) to
    Math (theta=azimuth, phi=polar).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.math_sph3d)
    {'r': Q(1., 'm'), 'theta': Angle(60, 'deg'), 'phi': Angle(30, 'deg')}

    >>> p = {"r": 1.0, "theta": 30, "phi": 60}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.math_sph3d, usys=usys)
    {'r': 1.0, 'theta': 60, 'phi': 30}

    """
    del usys  # Unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    return canonical_containers(
        {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}, to_chart
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: MathSpherical3D,
    to_M: Rn,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """MathSpherical3D -> Spherical3D.

    Swaps theta and phi: Math (theta=azimuth, phi=polar) to
    Physics (theta=polar, phi=azimuth).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Angle(30, 'deg'), 'phi': Angle(60, 'deg')}

    >>> p = {"r": 1.0, "theta": 60, "phi": 30}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.sph3d, usys=usys)
    {'r': 1.0, 'theta': 30, 'phi': 60}

    """
    del usys  # Unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    return canonical_containers(
        {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}, to_chart
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: ProlateSpheroidal3D,
    to_M: Rn,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""ProlateSpheroidal3D -> Cylindrical3D.

    Uses the focal length $\Delta$ stored on ``from_chart``.

    Validity constraints (enforced by the representation) are:

    - $\Delta > 0$,
    - $\mu \ge \Delta^2$,
    - $\lvert\nu\rvert \le \Delta^2$.

    The conversion proceeds via

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{\lvert\nu\rvert}{\Delta^2}\right)}$,
    $z = \sqrt{\mu\,\frac{\lvert\nu\rvert}{\Delta^2}}\,\mathrm{sign}(\nu)$,
    $\phi = \phi$.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cyl3d)
    {'rho': Q(0.8660254, 'm'), 'phi': Angle(0, 'rad'), 'z': Q(1.11803399, 'm')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cyl3d, usys=usys)
    {'rho': Array(0.8660254, dtype=float64), 'phi': 0,
     'z': Array(1.11803399, dtype=float64)}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    (mu, nu), unit_area = strip(p, ("mu", "nu"))
    delta2 = _delta_squared(from_chart, unit_area, usys)

    nu_d2 = jnp.abs(nu) / delta2
    # `mu` and `nu` are *areas*, so a length output carries the square root of
    # their unit. `phi` is an angle, outside the group, and passes through.
    unit_len = None if unit_area is None else unit_area**0.5
    return canonical_containers(
        {
            "rho": wrap(jnp.sqrt((mu - delta2) * (1 - nu_d2)), unit_len),
            "phi": p["phi"],
            "z": wrap(jnp.sqrt(mu * nu_d2) * jnp.sign(nu), unit_len),
        },
        to_chart,
    )


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cylindrical3D,
    to_M: Rn,
    to_chart: ProlateSpheroidal3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""Cylindrical3D -> ProlateSpheroidal3D.

    Uses the focal length $\Delta$ stored on ``to_chart``.

    Let $R^2 = \rho^2$ and $z^2 = z^2$ and define

    $S = R^2 + z^2 + \Delta^2$,
    $D_f = R^2 + z^2 - \Delta^2$,
    $D = \sqrt{D_f^2 + 4 R^2 \Delta^2}$.

    Then

    $\mu = \Delta^2 + \tfrac12(D + D_f)$ (with numerically-stable branches),
    $\lvert\nu\rvert = \dfrac{2\Delta^2}{S + D}\,z^2$,
    and $\nu = \lvert\nu\rvert\,\mathrm{sign}(z)$, with a stability fix when
    $\Delta^2 - \lvert\nu\rvert$ is small.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, prolatesph3d)
    {'mu': Q(9., 'm2'), 'nu': Q(4., 'm2'), 'phi': Angle(0, 'rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": u.Q(2.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(0.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, prolatesph3d)
    {'mu': Q(8., 'm2'), 'nu': Q(0., 'm2'), 'phi': Angle(0, 'rad')}

    Without units:

    >>> p = {"rho": 2.0, "phi": 0, "z": 3.0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, prolatesph3d, usys=usys)
    {'mu': Array(14.52079729, dtype=float64),
     'nu': Array(2.47920271, dtype=float64), 'phi': 0}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    # Pre-compute common terms
    R2 = p["rho"] ** 2
    z2 = p["z"] ** 2
    if not isinstance(R2, ABCQ) or not isinstance(z2, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'rho' or 'z', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = cast("Array", u.ustrip(usys["length"], to_chart.Delta)) ** 2
    else:
        Delta2 = plum.convert(to_chart.Delta**2, u.Q)

    sum_ = R2 + z2 + Delta2
    diff_ = R2 + z2 - Delta2

    # D = sqrt((R^2 + z^2 - Delta^2)^2 + 4 R^2 Delta^2)
    D = jnp.sqrt(diff_**2 + 4 * R2 * Delta2)

    # Handle special cases for z=0 or rho=0
    D = jnp.where(p["z"] == 0, sum_, D)
    D = jnp.where(p["rho"] == 0, jnp.abs(diff_), D)

    # Numerically stable branches (avoid dividing by small numbers)
    pos_mu_minus_delta = 0.5 * (D + diff_)
    pos_delta_minus_nu = Delta2 * R2 / pos_mu_minus_delta

    neg_delta_minus_nu = 0.5 * (D - diff_)
    neg_mu_minus_delta = Delta2 * R2 / neg_delta_minus_nu

    mu_minus_delta = jnp.where(diff_ >= 0, pos_mu_minus_delta, neg_mu_minus_delta)
    delta_minus_nu = jnp.where(diff_ >= 0, pos_delta_minus_nu, neg_delta_minus_nu)

    mu = Delta2 + mu_minus_delta

    # |nu| = 2 Delta^2 / (sum_ + D) * z^2
    abs_nu = 2 * Delta2 / (sum_ + D) * z2

    # Stability fix when Delta^2 - |nu| is small
    abs_nu = jnp.where(abs_nu * 2 > Delta2, Delta2 - delta_minus_nu, abs_nu)

    nu = abs_nu * jnp.sign(p["z"])

    return canonical_containers({"mu": mu, "nu": nu, "phi": p["phi"]}, to_chart)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D,
    to_M: Rn,
    to_chart: ProlateSpheroidal3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Cylindrical3D -> ProlateSpheroidal3D.

    ``ProlateSpheroidal3D`` is only registered as a target from ``Cylindrical3D``;
    without this route the generic ``A -> A.cartesian -> B`` fallback would send
    ``Cart3D -> Cart3D -> ProlateSpheroidal3D`` and recurse forever. Route through
    ``Cylindrical3D`` instead (mirrors the ``Cart3D -> AbstractSpherical3D`` rule).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, prolate)
    {'mu': Q(9., 'm2'), 'nu': Q(4., 'm2'), 'phi': Angle(0., 'rad')}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    cyl = Cylindrical3D(M=from_chart.M)
    p_cyl = cxcapi.pt_map(p, from_M, from_chart, to_M, cyl, usys=usys)
    out = cxcapi.pt_map(p_cyl, from_M, cyl, to_M, to_chart, usys=usys)
    return canonical_containers(cast("CDict", out), to_chart)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: ProlateSpheroidal3D,
    to_M: Rn,
    to_chart: ProlateSpheroidal3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""{class}`coordinax.charts.ProlateSpheroidal3D` -> itself.

    If the focal length is unchanged (``to_chart.Delta == from_chart.Delta``), this
    is the identity map.

    If the focal length changes, we convert via cylindrical coordinates:

    ``Prolate(Delta_in) -> Cylindrical -> Prolate(Delta_out)``.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Same focal length (identity transformation):

    >>> prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolate, cxm.R3, prolate)
    {'mu': Q(5., 'm2'), 'nu': Q(1., 'm2'), 'phi': Angle(0., 'rad')}

    Different focal lengths (converts via cylindrical):

    >>> prolate_in = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> prolate_out = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(3.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolate_in, cxm.R3, prolate_out)
    {'mu': Q(9.85889894, 'm2'), 'nu': Q(1.14110106, 'm2'), 'phi': Angle(0., 'rad')}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    # Cast to the result type
    dtype = jnp.result_type(
        to_chart.Delta, from_chart.Delta, *[v.dtype for v in p.values()]
    )
    p = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), p)
    cyl3d = Cylindrical3D(M=to_chart.M)
    # `Delta == Delta` is a *dimensionless Quantity* whenever either side is a
    # `unxt.Quantity`, and `lax.cond` rejects that as a predicate. Compare in a
    # common unit and strip to a plain array bool, valid for a static and a
    # traced `Delta` alike. Cost: a `Delta` of the wrong dimension now raises
    # `UnitConversionError` here, instead of taking the conversion branch.
    unit = from_chart.Delta.unit
    same_delta = u.ustrip(unit, to_chart.Delta) == u.ustrip(unit, from_chart.Delta)

    # Both branches must agree on pytree *structure*, and `Angle` and `Quantity`
    # are different nodes, so the pass-through branch canonicalises too. Without
    # it the converting branch returns `Angle` while this one hands back
    # whatever the caller built, and `lax.cond` rejects the mismatched pair.
    def keep(p: CDict) -> CDict:
        return canonical_containers(p, to_chart)

    def convert(p: CDict) -> CDict:
        out = cxcapi.pt_map(p, from_M, from_chart, to_M, cyl3d, usys=usys)
        return cast(
            "CDict", cxcapi.pt_map(out, from_M, cyl3d, to_M, to_chart, usys=usys)
        )

    # `lax.cond` traces *both* branches, so routing a same-`Delta` identity
    # through it costs a full round-trip to cylindrical and back -- ~65ms
    # eagerly, the same as actually converting. A `Delta` that is concrete
    # (the `StaticQuantity` every ordinary call site passes, or a dynamic one
    # outside `jit`) can decide in Python and skip the dead branch entirely.
    # A traced `Delta` cannot, and still needs `lax.cond`; that path is what
    # `test_prolate_differentiable` covers.
    if isinstance(same_delta, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
        return jax.lax.cond(same_delta, keep, convert, p)
    return keep(p) if bool(same_delta) else convert(p)  # concrete 0-d array


# -----------------------------------------------
# N-D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: CartND,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """CartND -> AbstractChart.

    Converts from N-dimensional Cartesian (with a single 'q' array) to any
    other chart type by first extracting the appropriate fixed-dimensional
    Cartesian representation.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Convert 3D CartND to Spherical:

    >>> p = {"q": u.Q([1.0, 0.0, 0.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    Convert 2D CartND to Polar:

    >>> p = {"q": u.Q([3.0, 4.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R2, cxc.polar2d)
    {'r': Q(5., 'm'), 'theta': Angle(0.92729522, 'rad')}

    Convert 1D CartND to Radial:

    >>> p = {"q": u.Q([5.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R1, cxc.radial1d)
    {'r': Q(5., 'm')}

    Convert CartND to Cart3D:

    >>> p = {"q": u.Q([1.0, 2.0, 3.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R3, cxc.cart3d)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    if from_chart.M not in (from_M, to_chart.M) or to_M != to_chart.M:
        msg = (
            f"no transition from {from_chart} on {from_M} to {to_chart} on "
            f"{to_M}: the manifolds do not line up"
        )
        raise ManifoldMismatchError(msg)

    # If target is CartND, we can't convert (would be infinite recursion)
    if isinstance(to_chart, CartND):
        msg = "Cannot convert CartND to CartND via this dispatch."
        raise TypeError(msg)

    # Get the corresponding fixed-dimensional Cartesian chart
    cart_chart = to_chart.cartesian

    # If cartesian_chart returns CartND, we don't support this conversion
    if isinstance(cart_chart, CartND):
        msg = f"CartND conversion not supported for {type(to_chart).__name__}."
        raise NotImplementedError(msg)

    # Get dimensionality from the target chart
    target_ndim = to_chart.ndim

    # Check that the CartND data has the right dimensionality. Components are on
    # the last axis (leading axes are batch), matching the `q[..., i]` unpack below.
    q = p["q"]
    data_ndim = q.shape[-1]
    if data_ndim != target_ndim:
        msg = (
            f"CartND data has {data_ndim} dimensions but target chart "
            f"{type(to_chart).__name__} requires {target_ndim} dimensions."
        )
        raise ValueError(msg)

    # Convert CartND to fixed-dimensional Cartesian
    p_cart = {k: q[..., i] for i, k in enumerate(cart_chart.components)}

    # If target is already the Cartesian chart, return directly
    if type(to_chart) is type(cart_chart):
        return canonical_containers(p_cart, to_chart)

    # Otherwise, transform from Cartesian to target chart
    out = cxcapi.pt_map(p_cart, cart_chart.M, cart_chart, to_M, to_chart, usys=usys)
    return canonical_containers(cast("CDict", out), to_chart)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: CartND,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> CartND.

    Converts from any chart type to N-dimensional Cartesian (with a single
    'q' array) by first transforming to the appropriate fixed-dimensional
    Cartesian representation.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Convert Cart3D to CartND:

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.cartnd)
    {'q': Q([1., 2., 3.], 'm')}

    Convert Cart2D to CartND:

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> cxc.pt_map(p, cxm.R2, cxc.cart2d, cxm.R2, cxc.cartnd)
    {'q': Q([3., 4.], 'm')}

    Convert Radial to CartND:

    >>> p = {"r": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxc.radial1d, cxc.cartnd)
    {'q': Q([3.], 'm')}

    Convert Cylindrical to CartND (z-axis point):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxc.cyl3d, cxc.cartnd)
    {'q': Q([0., 0., 5.], 'm')}

    """
    if (
        from_M != from_chart.M
        or to_chart.M not in (from_M, RN)
        or to_M not in (from_M, RN)
    ):
        msg = (
            f"no transition from {from_chart} on {from_M} to {to_chart} on "
            f"{to_M}: the manifolds do not line up"
        )
        raise ManifoldMismatchError(msg)

    # If source is CartND, we can't convert (would be infinite recursion)
    if isinstance(from_chart, CartND):
        msg = "Cannot convert CartND to CartND via this dispatch."
        raise TypeError(msg)

    # Get the corresponding fixed-dimensional Cartesian chart
    cart_chart = from_chart.cartesian

    # If cartesian_chart returns CartND, we don't support this conversion
    if isinstance(cart_chart, CartND):
        msg = f"CartND conversion not supported for {type(from_chart).__name__}."
        raise NotImplementedError(msg)

    # Transform from source to fixed-dimensional Cartesian
    p_cart = cxcapi.pt_map(p, from_M, from_chart, from_M, cart_chart, usys=usys)
    p_cart = cast("dict[str, Array]", p_cart)

    # Convert fixed-dimensional Cartesian to CartND
    q = jnp.stack([p_cart[k] for k in cart_chart.components], axis=-1)

    return canonical_containers({"q": q}, to_chart)


# ===================================================================
# Point Transform a Quantity
# Only quantities which have the same units for all components can be
# transformed as a single Quantity.


@plum.dispatch.multi(
    *(
        (u.AbstractQuantity, EuclideanManifold, typ, EuclideanManifold, typ)
        for typ in (Cart0D, Cart1D, Radial1D, Time1D, Cart2D, Cart3D, CartND)
    )
)
def pt_map(
    q: u.AbstractQuantity,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> u.AbstractQuantity:
    """Identity point transform for Quantity inputs on uniform-unit charts.

    For charts where all components share the same unit (Cartesian charts, 0D/1D
    charts), a Quantity can be passed directly and is returned unchanged when
    the source and target charts are the same type.

    This dispatch only handles identity transformations (same chart type).  For
    transformations between different chart types with Quantity input, the
    Quantity must first be converted to a coordinate dictionary.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    **1D Cartesian (identity):**

    >>> q = u.Q([5.0], "m")
    >>> cxc.pt_map(q, cxm.R1, cxc.cart1d, cxm.R1, cxc.cart1d, usys=None) is q
    True

    **2D Cartesian (identity):**

    >>> q = u.Q([3.0, 4.0], "m")
    >>> cxc.pt_map(q, cxm.R2, cxc.cart2d, cxm.R2, cxc.cart2d, usys=None) is q
    True

    **3D Cartesian (identity):**

    >>> q = u.Q([1.0, 2.0, 3.0], "km")
    >>> cxc.pt_map(q, cxm.R3, cxc.cart3d, cxm.R3, cxc.cart3d, usys=None) is q
    True

    **N-D Cartesian (identity):**

    >>> q = u.Q([1.0, 2.0, 3.0, 4.0], "m")
    >>> cxc.pt_map(q, cxm.RN, cxc.cartnd, cxm.RN, cxc.cartnd, usys=None) is q
    True

    """
    del usys  # Unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    return q


@plum.dispatch
def pt_map(
    p: u.AbstractQuantity,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> ul.QuantityMatrix:
    """Transform a QuantityMatrix between charts.

    Converts the components of a QuantityMatrix from one chart to another,
    preserving the matrix structure with potentially different units per component.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    **2D Cartesian to Polar:**

    >>> q = u.Q([3.0, 4.0], "m")
    >>> result = cxc.pt_map(q, cxc.cart2d, cxc.polar2d)
    >>> result.shape
    (2,)
    >>> result.unit
    UnitsMatrix("(m, rad)")

    **3D Cartesian to Spherical:**

    >>> q = u.Q([1.0, 0.0, 0.0], "kpc")
    >>> result = cxc.pt_map(q, cxc.cart3d, cxc.sph3d)
    >>> result.shape
    (3,)

    **Batched transformation:**

    >>> q_batch = u.Q([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "m")
    >>> result = cxc.pt_map(q_batch, cxc.cart3d, cxc.sph3d)
    >>> result.shape
    (2, 3)

    """
    # Build a dict of arrays for each component
    p_dict = cxcapi.cdict(p, from_chart)

    # Transform the point dict
    p_to = cxcapi.pt_map(p_dict, from_M, from_chart, to_M, to_chart, usys=usys)
    p_to = cast("dict[str, u.AbstractQuantity]", p_to)

    # Stack the transformed components into an QuantityMatrix
    p_out = ul.QuantityMatrix(
        jnp.stack([u.ustrip(p_to[k]) for k in to_chart.components], axis=-1),
        unit=ul.UnitsMatrix(ul.cdict_units(p_to, to_chart.components)),
    )

    return p_out  # noqa: RET504


# ===================================================================
# Point Transform an Array


@plum.dispatch
def pt_map(
    p: Array | list,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys,
) -> Array:
    r"""Point transform for array input.

    Transforms a point represented as a raw array (without units) from one chart
    to another. The unit system ``usys`` provides the units for interpreting the
    array components.

    Returns
    -------
    Array
        Array of shape ``(..., ndim)`` containing the transformed coordinates in
        ``to_chart``.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    **Cartesian to Spherical (3D):**

    >>> usys = u.unitsystem("m", "rad")
    >>> p = jnp.array([1.0, 0.0, 0.0])  # Point on x-axis
    >>> cxc.pt_map(p, cxc.cart3d, cxc.sph3d, usys=usys)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    The result is [r, theta, phi] = [1, pi/2, 0] (on equator, x-axis).

    **Spherical to Cartesian (3D):**

    >>> p = jnp.array([2.0, jnp.pi/4, 0.0])  # r=2, theta=45°, phi=0
    >>> cxc.pt_map(p, cxc.sph3d, cxc.cart3d, usys=usys)
    Array([1.41421356, 0.        , 1.41421356], dtype=float64)

    **Cartesian to Cylindrical:**

    >>> p = jnp.array([3.0, 4.0, 5.0])
    >>> cxc.pt_map(p, cxc.cart3d, cxc.cyl3d, usys=usys)
    Array([5.        , 0.92729522, 5.        ], dtype=float64)

    The result is [rho, phi, z] = [5, arctan(4/3), 5].

    **Batched transformation:**

    >>> p_batch = jnp.array([[1.0, 0.0, 0.0],
    ...                      [0.0, 1.0, 0.0],
    ...                      [0.0, 0.0, 1.0]])
    >>> cxc.pt_map(p_batch, cxc.cart3d, cxc.sph3d, usys=usys)
    Array([[1.        , 1.57079633, 0.        ],
           [1.        , 1.57079633, 1.57079633],
           [1.        , 0.        , 0.        ]], dtype=float64)

    **2D Cartesian to Polar:**

    >>> usys_2d = u.unitsystem("m", "rad")
    >>> p = jnp.array([3.0, 4.0])
    >>> cxc.pt_map(p, cxc.cart2d, cxc.polar2d, usys=usys_2d)
    Array([5.        , 0.92729522], dtype=float64)

    """
    if usys is None:
        msg = "usys must be provided for array input."
        raise ValueError(msg)

    # Build a dict of arrays for each component
    p_dict = cxcapi.cdict(jnp.asarray(p), from_chart)

    # Transform the point dict
    p_to = cxcapi.pt_map(p_dict, from_M, from_chart, to_M, to_chart, usys=usys)
    p_to = cast("dict[str, Array]", p_to)

    # Stack the transformed components into an array
    p_out: Array = jnp.stack([p_to[comp] for comp in to_chart.components], axis=-1)

    return p_out


# ===================================================================
# Cartesian phase space (cart3d x cart3d) -> Poincaré symplectic polar


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: CartesianProductManifold,
    from_chart: CartesianProductChart,
    to_M: NoManifold,
    to_chart: PoincarePolar6D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""Cartesian phase space ``cart3d x cart3d`` -> ``PoincarePolar6D`` (gala forward).

    The source is a two-factor Cartesian product chart: factor 0 is position
    ``(x, y, z)``, factor 1 its velocity ``(vx, vy, vz)``. Implements gala's
    ``cartesian_to_poincare_polar`` (Papaphilippou & Laskar 1996):

    ``rho = hypot(x, y)``,  ``phi = atan2(x, y)``  (gala's azimuth convention),
    ``dt_rho = (x*vx + y*vy) / rho``,  ``Lz = x*vy - y*vx``,
    ``pp_phi = sqrt(2|Lz|) cos(phi)``,  ``pp_phidot = sqrt(2|Lz|) sin(phi)``,
    ``dt_z = vz``.

    ``sqrt(|Lz|)`` discards ``sign(Lz)``, so there is no *global* inverse. A
    *partial* inverse (assuming ``Lz >= 0``) is registered below.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> ps = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
    >>> q = {"q.x": u.Q(3.0, "kpc"), "q.y": u.Q(4.0, "kpc"), "q.z": u.Q(5.0, "kpc"),
    ...      "p.x": u.Q(1.0, "kpc/Myr"), "p.y": u.Q(2.0, "kpc/Myr"),
    ...      "p.z": u.Q(0.5, "kpc/Myr")}
    >>> out = cxc.pt_map(q, ps.M, ps, cxc.poincarepolar6d.M, cxc.poincarepolar6d)
    >>> sorted(out)
    ['dt_rho', 'dt_z', 'pp_phi', 'pp_phidot', 'rho', 'z']

    Lz = x*vy - y*vx = 2 kpc^2/Myr, so sqrt(2|Lz|) = 2; phi = atan2(3, 4):

    >>> out["rho"], out["dt_rho"], out["dt_z"]
    (Q(5., 'kpc'), Q(2.2, 'kpc / Myr'), Q(0.5, 'kpc / Myr'))
    >>> out["pp_phi"].round(4), out["pp_phidot"].round(4)
    (Q(1.6, 'kpc / Myr(1/2)'), Q(1.2, 'kpc / Myr(1/2)'))

    A non-Cartesian or wrong-arity product source is rejected:

    >>> bad = cxc.CartesianProductChart((cxc.cart3d, cxc.polar2d), ("q", "p"))
    >>> try:
    ...     cxc.pt_map({}, bad.M, bad, cxc.poincarepolar6d.M, cxc.poincarepolar6d)
    ... except NotImplementedError:
    ...     print("rejected")
    rejected

    """
    del usys
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    _require_cart3d_phase_space(from_chart, direction="to")

    pos, vel = from_chart.split_components(p)
    # Two dimensional groups: the positions are lengths, the velocities are
    # velocities. `z` and `vz` are carried through untouched -- the arithmetic
    # never reads them -- so they keep their own units.
    (x, y), unit_len = strip(pos, ("x", "y"))
    (vx, vy), unit_vel = strip(vel, ("x", "y"))

    rho = jnp.hypot(x, y)
    phi = jnp.arctan2(x, y)  # gala convention: azimuth from +y
    lz = x * vy - y * vx
    s = jnp.sqrt(2 * jnp.abs(lz))
    # On the axis (rho == 0) the numerator x*vx + y*vy is also 0; define
    # dt_rho == 0 there by convention instead of 0/0 -> NaN.
    dt_rho = _ratio_zero_on_axis(x * vx + y * vy, rho)
    # `lz` is a length times a velocity, so `s = sqrt(2|lz|)` carries the
    # square root of that product -- `kpc / Myr(1/2)` for kpc and kpc/Myr.
    unit_s = (
        None if unit_len is None or unit_vel is None else (unit_len * unit_vel) ** 0.5
    )
    return {
        "rho": wrap(rho, unit_len),
        "pp_phi": wrap(s * jnp.cos(phi), unit_s),
        "z": pos["z"],
        "dt_rho": wrap(dt_rho, unit_vel),
        "pp_phidot": wrap(s * jnp.sin(phi), unit_s),
        "dt_z": vel["z"],
    }


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: NoManifold,
    from_chart: PoincarePolar6D,
    to_M: CartesianProductManifold,
    to_chart: CartesianProductChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""``PoincarePolar6D`` -> Cartesian phase space (partial inverse of gala map).

    Inverts the gala forward map. Because the forward uses ``sqrt(2|Lz|)`` the
    sign of the angular momentum is not recoverable, so this assumes ``Lz >= 0``
    (the standard convention) and is a *partial* inverse — exact only when the
    original point had non-negative ``Lz``:

    ``s = hypot(pp_phi, pp_phidot)``,  ``phi = atan2(pp_phidot, pp_phi)``,
    ``Lz = s**2 / 2``,  ``x = rho sin(phi)``,  ``y = rho cos(phi)``,
    ``vx = sin(phi) dt_rho - cos(phi) Lz/rho``,
    ``vy = cos(phi) dt_rho + sin(phi) Lz/rho``,  ``vz = dt_z``.

    (Singular on the axis ``rho = 0``, inherent to the coordinates.)

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> ps = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))

    Round-trips a forward result whose Lz >= 0:

    >>> q = {"q.x": u.Q(3.0, "kpc"), "q.y": u.Q(4.0, "kpc"), "q.z": u.Q(5.0, "kpc"),
    ...      "p.x": u.Q(1.0, "kpc/Myr"), "p.y": u.Q(2.0, "kpc/Myr"),
    ...      "p.z": u.Q(0.5, "kpc/Myr")}
    >>> pp = cxc.pt_map(q, ps.M, ps, cxc.poincarepolar6d.M, cxc.poincarepolar6d)
    >>> back = cxc.pt_map(pp, cxc.poincarepolar6d.M, cxc.poincarepolar6d, ps.M, ps)
    >>> back["q.x"].round(6), back["q.y"].round(6), back["p.x"].round(6)
    (Q(3., 'kpc'), Q(4., 'kpc'), Q(1., 'kpc / Myr'))

    """
    del usys
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    _require_cart3d_phase_space(to_chart, direction="from")

    # `z` and `dt_z` are carried through untouched; the arithmetic never reads
    # them, so they keep their own units.
    (rho,), unit_rho = strip(p, ("rho",))
    (dt_rho,), unit_vel = strip(p, ("dt_rho",))
    (pp_phi, pp_phidot), unit_s = strip(p, ("pp_phi", "pp_phidot"))

    phi = jnp.arctan2(pp_phidot, pp_phi)
    # Lz = s²/2 with s = hypot(pp_phi, pp_phidot); compute directly to skip the
    # sqrt (sign(Lz) is not recoverable from the forward map, so take Lz >= 0).
    lz = (pp_phi**2 + pp_phidot**2) / 2
    sinp, cosp = jnp.sin(phi), jnp.cos(phi)

    # On the axis (rho == 0, where lz == 0 too) define lz/rho == 0 by convention.
    lz_over_rho = _ratio_zero_on_axis(lz, rho)
    # `lz` carries a length times a velocity and `rho` a length, so the ratio is
    # a velocity -- but in `unit_s**2 / unit_rho`, which need not be the unit
    # `dt_rho` came in. The `Quantity` form converted it on the subtraction
    # below; do the same explicitly so both terms are in `dt_rho`'s unit.
    if unit_s is not None and unit_rho is not None and unit_vel is not None:
        lz_over_rho = u.uconvert_value(unit_vel, unit_s**2 / unit_rho, lz_over_rho)

    pos = {
        "x": wrap(rho * sinp, unit_rho),
        "y": wrap(rho * cosp, unit_rho),
        "z": p["z"],
    }
    vel = {
        "x": wrap(sinp * dt_rho - cosp * lz_over_rho, unit_vel),
        "y": wrap(cosp * dt_rho + sinp * lz_over_rho, unit_vel),
        "z": p["dt_z"],
    }
    return canonical_containers(to_chart.merge_components((pos, vel)), to_chart)
