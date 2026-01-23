"""Conversion functions for vector charts."""
from jaxtyping import ArrayLike, Array

__all__: tuple[str, ...] = ()


from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as ABCQ  # noqa: N814

import coordinax._src.charts as cxc
import coordinax._src.embed as cxe
import coordinax._src.roles as cxr
from coordinax._src import api
from coordinax._src.custom_types import ComponentsKey, CsDict, OptUSys
from coordinax._src.utils import uconvert_to_rad

# ===================================================================
# Support for the higher-level `vconvert` function


@plum.dispatch
def vconvert(
    role: cxr.Point,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    *_: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    """Convert a point from one representation to another.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.roles as cxr
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    **Cartesian to Spherical (with units):**

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cx.vconvert(cxr.point, cxc.sph3d, cxc.cart3d, p)
    {'r': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    **Cylindrical to Cartesian (without units):**

    >>> p = {"rho": 3.0, "phi": 0, "z": 4.0}
    >>> cx.vconvert(cxr.point, cxc.cart3d, cxc.cyl3d, p)
    {'x': Array(3., dtype=float64, ...), 'y': Array(0., dtype=float64, ...),
     'z': 4.0}

    **Polar to Cartesian (2D):**

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cx.vconvert(cxr.point, cxc.cart2d, cxc.polar2d, p)
    {'x': Quantity(Array(3...e-16, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(5., dtype=float64, ...), unit='m')}

    **Between Spherical variants (Spherical to LonLatSpherical):**

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> cx.vconvert(cxr.point, cxc.lonlatsph3d, cxc.sph3d, p)
    {'lon': Quantity(Array(0, dtype=int64, ...), unit='deg'),
     'lat': Quantity(Array(45., dtype=float64, ...), unit='deg'),
     'distance': Quantity(Array(1., dtype=float64, ...), unit='m')}

    **Identity conversion (same chart):**

    >>> p = {"x": u.Q(2.0, "m"), "y": u.Q(3.0, "m")}
    >>> cx.vconvert(cxr.point, cxc.cart2d, cxc.cart2d, p) is p
    True

    """
    # Point transforms via point_transform (no tangent/base point needed)
    return api.point_transform(to_chart, from_chart, p, usys=usys)


# ===================================================================
# Self representation conversions


@plum.dispatch.multi(
    *(
        (typ, typ, dict[ComponentsKey, Any])
        for typ in [
            # 0D
            cxc.Cart0D,
            # 1D
            cxc.Cart1D,
            cxc.Radial1D,
            cxc.Time1D,
            # 2D
            cxc.Cart2D,
            cxc.Polar2D,
            cxc.TwoSphere,
            # 3D
            cxc.Cart3D,
            cxc.Cylindrical3D,
            cxc.Spherical3D,
            cxc.LonLatSpherical3D,
            cxc.LonCosLatSpherical3D,
            cxc.MathSpherical3D,
            # cxc.ProlateSpheroidal3D,  # requires Delta
            # 4D
            # SpaceTimeCT,  # depends on spatial representation
            # 6D
            cxc.PoincarePolar6D,
            # N-D
            cxc.CartND,
        ]
    )
)
def point_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CsDict,  # type: ignore[type-arg]
    /,
    usys: OptUSys = None,
) -> CsDict:  # type: ignore[type-arg]
    """Identity conversion for matching representations.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> p2 = cxt.point_transform(cxc.cart2d, cxc.cart2d, p)
    >>> p is p2
    True

    >>> p = {"r": u.Q(3.0, "m")}
    >>> p2 = cxt.point_transform(cxc.radial1d, cxc.radial1d, p)
    >>> p is p2
    True

    """
    return p


# ---------------------------------------------------------
# General representation conversions


@plum.dispatch(precedence=-1)
def point_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """AbstractChart -> Cartesian -> AbstractChart."""
    p_cart = api.point_transform(to_chart.cartesian, from_chart, p, usys=usys)
    return api.point_transform(to_chart, from_chart.cartesian, p_cart, usys=usys)


# ---------------------------------------------------------
# Specific representation conversions

# -----------------------------------------------
# 1D


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart1D, from_chart: cxc.Radial1D, p: CsDict, /, usys: OptUSys = None
) -> CsDict:
    """Radial1D -> Cartesian1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m")}
    >>> cxt.point_transform(cxc.cart1d, cxc.radial1d, p)
    {'x': Quantity(Array(5., dtype=float64, ...), unit='m')}

    >>> p = {"r": 5.0}  # No units
    >>> cxt.point_transform(cxc.cart1d, cxc.radial1d, p)
    {'x': 5.0}

    """
    del to_chart, from_chart, usys
    return {"x": p["r"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.Radial1D, from_chart: cxc.Cart1D, p: CsDict, /, usys: OptUSys = None
) -> CsDict:
    """Cartesian1D -> Radial1D.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"x": u.Q(5.0, "m")}
    >>> cxt.point_transform(cxc.radial1d, cxc.cart1d, p)
    {'r': Quantity(Array(5., dtype=float64, ...), unit='m')}

    >>> p = {"x": 5.0}  # No units
    >>> cxt.point_transform(cxc.radial1d, cxc.cart1d, p)
    {'r': 5.0}

    """
    del to_chart, from_chart, usys  # Unused
    return {"r": p["x"]}


# -----------------------------------------------
# 2D


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart2D, from_chart: cxc.Polar2D, p: CsDict, /, usys: OptUSys = None
) -> CsDict:
    """Polar2D -> Cart2D.

    The `r` and `theta` coordinates are converted to the `x` and `y` coordinates
    of the 2D Cartesian system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxt.point_transform(cxc.cart2d, cxc.polar2d, p)
    {'x': Quantity(Array(3...e-16, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(5., dtype=float64, ...), unit='m')}

    >>> p = {"r": 5, "theta": 90}  # No units
    >>> usys = u.unitsystem("km", "deg")
    >>> cxt.point_transform(cxc.cart2d, cxc.polar2d, p, usys=usys)
    {'x': Array(3.061617e-16, dtype=float64, ...), 'y': Array(5., dtype=float64, ...)}

    """
    del to_chart, from_chart  # Unused
    theta = uconvert_to_rad(p["theta"], usys)
    x = p["r"] * jnp.cos(theta)
    y = p["r"] * jnp.sin(theta)
    return {"x": x, "y": y}


@plum.dispatch
def point_transform(
    to_chart: cxc.Polar2D, from_chart: cxc.Cart2D, p: CsDict, /, usys: OptUSys = None
) -> CsDict:
    """Cart2D -> Polar2D.

    The `x` and `y` coordinates are converted to the `r` and `theta` coordinates
    of the 2D polar system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"x": u.Q(3, "m"), "y": u.Q(4, "m")}
    >>> cxt.point_transform(cxc.polar2d, cxc.cart2d, p)
    {'r': Quantity(Array(5., dtype=float64, ...), unit='m'),
    'theta': Quantity(Array(0.92729522, dtype=float64, ...), unit='rad')}

    >>> p = {"x": 3, "y": 4}  # No units
    >>> cxt.point_transform(cxc.polar2d, cxc.cart2d, p)
    {'r': Array(5., dtype=float64, ...),
     'theta': Array(0.92729522, dtype=float64, ...)}

    """
    del to_chart, from_chart, usys  # Unused
    r_ = jnp.hypot(p["x"], p["y"])
    theta = jnp.arctan2(p["y"], p["x"])
    return {"r": r_, "theta": theta}


# -----------------------------------------------
# 3D


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart3D,
    from_chart: cxc.Cylindrical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Cylindrical3D -> Cart3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"rho": u.Q(1.0, "m"), "phi": u.Q(90, "deg"), "z": u.Q(2.0, "m")}
    >>> cxt.point_transform(cxc.cart3d, cxc.cyl3d, p)
    {'x': Quantity(Array(6.123234e-17, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(2., dtype=float64, ...), unit='m')}

    >>> p = {"rho": 1.0, "phi": 90, "z": 2.0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.cart3d, cxc.cyl3d, p, usys=usys)
    {'x': Array(6...e-17, dtype=float64, ...),
     'y': Array(1., dtype=float64, ...),
     'z': 2.0}

    """
    del to_chart, from_chart  # Unused
    phi = uconvert_to_rad(p["phi"], usys)
    x = p["rho"] * jnp.cos(phi)
    y = p["rho"] * jnp.sin(phi)
    return {"x": x, "y": y, "z": p["z"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart3D,
    from_chart: cxc.Spherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Spherical3D -> Cart3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxt.point_transform(cxc.cart3d, cxc.sph3d, p)
    {'x': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(1., dtype=float64, ...), unit='m')}

    A point on the equator (theta=90 deg, phi=0):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.cart3d, cxc.sph3d, p, usys=usys)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    del to_chart, from_chart  # Unused
    r_ = p["r"]
    theta = uconvert_to_rad(p["theta"], usys)
    phi = uconvert_to_rad(p["phi"], usys)
    x = r_ * jnp.sin(theta) * jnp.cos(phi)
    y = r_ * jnp.sin(theta) * jnp.sin(phi)
    z = r_ * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart3D,
    from_chart: cxc.LonLatSpherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """LonLatSpherical3D -> Cart3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    A point at the north pole (lat=90 deg):

    >>> p = {"lon": u.Q(0, "deg"), "lat": u.Q(90, "deg"), "distance": u.Q(1.0, "m")}
    >>> cxt.point_transform(cxc.cart3d, cxc.lonlatsph3d, p)
    {'x': Quantity(Array(6.123234e-17, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(1., dtype=float64, ...), unit='m')}

    A point on the equator at lon=0:

    >>> p = {"lon": 0, "lat": 0, "distance": 2}
    >>> cxt.point_transform(cxc.cart3d, cxc.lonlatsph3d, p)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(0., dtype=float64, ...)}

    """
    del to_chart, from_chart  # Unused
    r_ = p["distance"]
    lon = uconvert_to_rad(p["lon"], usys)
    lat = uconvert_to_rad(p["lat"], usys)
    x = r_ * jnp.cos(lat) * jnp.cos(lon)
    y = r_ * jnp.cos(lat) * jnp.sin(lon)
    z = r_ * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart3D,
    from_chart: cxc.LonCosLatSpherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """LonCosLatSpherical3D -> Cart3D.

    Components are (lon_coslat, lat, r), where lon_coslat := lon * cos(lat).
    Longitude is undefined at the poles (cos(lat) == 0); we set lon = 0 by
    convention there to avoid NaNs.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    A point on the equator (lat=0, so lon_coslat = lon):

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
    ...      "distance": u.Q(1.0, "m")}
    >>> cxt.point_transform(cxc.cart3d, cxc.loncoslatsph3d, p)
    {'x': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(0., dtype=float64, ...), unit='m')}

    At the north pole (lat=90), lon_coslat is effectively 0 regardless of lon:

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(90, "deg"),
    ...      "distance": u.Q(2.0, "m")}
    >>> cxt.point_transform(cxc.cart3d, cxc.loncoslatsph3d, p)
    {'x': Quantity(Array(1.2246468e-16, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(2., dtype=float64, ...), unit='m')}

    """
    del to_chart, from_chart
    lon_coslat, r_ = p["lon_coslat"], p["distance"]
    lat = uconvert_to_rad(p["lat"], usys)
    # Handle the poles where cos(lat) == 0
    coslat = jnp.cos(lat)
    lon = jnp.where(coslat == 0, 0, lon_coslat / coslat)
    lon = uconvert_to_rad(lon, usys)
    # Convert to Cartesian
    x = r_ * jnp.cos(lat) * jnp.cos(lon)
    y = r_ * jnp.cos(lat) * jnp.sin(lon)
    z = r_ * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart3D,
    from_chart: cxc.MathSpherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """MathSpherical3D -> Cart3D.

    - theta: azimuth in the x-y plane (longitude-like)
    - phi  : polar angle from +z, with phi in [0, pi]

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    A point on the +z axis (phi=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxt.point_transform(cxc.cart3d, cxc.mathsph3d, p)
    {'x': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(1., dtype=float64, ...), unit='m')}

    A point on the +x axis (theta=0, phi=90):

    >>> p = {"r": u.Q(2.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(90, "deg")}
    >>> cxt.point_transform(cxc.cart3d, cxc.mathsph3d, p)
    {'x': Quantity(Array(2., dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(1.2246468e-16, dtype=float64, ...), unit='m')}

    """
    del to_chart, from_chart  # Unused
    r_ = p["r"]
    theta = uconvert_to_rad(p["theta"], usys)
    phi = uconvert_to_rad(p["phi"], usys)
    x = r_ * jnp.sin(phi) * jnp.cos(theta)
    y = r_ * jnp.sin(phi) * jnp.sin(theta)
    z = r_ * jnp.cos(phi)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cart3D,
    from_chart: cxc.ProlateSpheroidal3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    r"""ProlateSpheroidal3D -> Cart3D.

    We calculate through cylindrical coordinates first:

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\;\mathrm{sign}(\nu)$
    $\phi = \phi.$

    Then convert to Cartesian:

    $x=\rho\cos\phi$, $y=\rho\sin\phi$, $z=z$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxt.point_transform(cxc.cart3d, prolatesph3d, p)
    {'x': Quantity(Array(0.8660254, dtype=float64), unit='m'),
     'y': Quantity(Array(0., dtype=float64), unit='m'),
     'z': Quantity(Array(1.11803399, dtype=float64), unit='m')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxt.point_transform(cxc.cart3d, prolatesph3d, p, usys=usys)
    {'x': Array(0.8660254, dtype=float64),
     'y': Array(0., dtype=float64),
     'z': Array(1.11803399, dtype=float64)}

    """
    # Calculate cylindrical distance
    nu, mu = p["nu"], p["mu"]
    if not isinstance(nu, ABCQ) or not isinstance(mu, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'mu' or 'nu', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = u.ustrip(usys["length"], from_chart.Delta) ** 2
    else:
        # TODO: fix Delta**2 issue in unxt
        Delta2 = u.Q.from_(from_chart.Delta) ** 2

    nu_D2 = jnp.abs(nu) / Delta2
    rho = jnp.sqrt((mu - Delta2) * (1 - nu_D2))
    # Convert to Cartesian
    phi = uconvert_to_rad(p["phi"], usys)
    x = rho * jnp.cos(phi)
    y = rho * jnp.sin(phi)
    z = jnp.sqrt(mu * nu_D2) * jnp.sign(nu)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cylindrical3D,
    from_chart: cxc.Cart3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Cart3D -> Cylindrical3D.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxt.point_transform(cxc.cyl3d, cxc.cart3d, p)
    {'rho': Quantity(Array(5., dtype=float64, ...), unit='m'),
     'phi': Quantity(Array(0.92729522, dtype=float64, ...), unit='rad'),
     'z': Quantity(Array(5., dtype=float64, ...), unit='m')}

    >>> p = {"x": 3.0, "y": 4.0, "z": 5.0}  # No units
    >>> cxt.point_transform(cxc.cyl3d, cxc.cart3d, p)
    {'rho': Array(5., dtype=float64, ...),
     'phi': Array(0.92729522, dtype=float64, ...),
     'z': 5.0}

    """
    del to_chart, from_chart, usys  # Unused
    rho = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"rho": rho, "phi": phi, "z": p["z"]}


@plum.dispatch.multi(
    (cxc.AbstractSpherical3D, cxc.Cart3D, Mapping),
    (cxc.AbstractSpherical3D, cxc.Cylindrical3D, Mapping),
)
def point_transform(
    to_chart: cxc.AbstractSpherical3D,
    from_chart: cxc.Cart3D | cxc.Cylindrical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Cart3D -> Spherical3D -> AbstractSpherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxt.point_transform(cxc.loncoslatsph3d, cxc.cart3d, p)
    {'lon_coslat': Quantity(Array(0., dtype=float64), unit='rad'),
     'lat': Quantity(Array(90., dtype=float64), unit='deg'),
     'distance': Quantity(Array(1., dtype=float64, weak_type=True), unit='m')}

    >>> p = {"rho": 0, "phi": 180, "z": 1}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.loncoslatsph3d, cxc.cyl3d, p, usys=usys)
    {'lon_coslat': Array(1.10218212e-14, dtype=float64),
     'lat': Array(1.57079633, dtype=float64),
     'distance': Array(1., dtype=float64, weak_type=True)}

    """
    p_sph = api.point_transform(cxc.sph3d, from_chart, p, usys=usys)
    return api.point_transform(to_chart, cxc.sph3d, p_sph, usys=usys)


@plum.dispatch
def point_transform(
    to_chart: cxc.Spherical3D,
    from_chart: cxc.Cart3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Cart3D -> Spherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    A point on the +z axis:

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxt.point_transform(cxc.sph3d, cxc.cart3d, p)
    {'r': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(0., dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    A point on the +x axis:

    >>> p = {"x": 2.0, "y": 0.0, "z": 0.0}  # No units
    >>> cxt.point_transform(cxc.sph3d, cxc.cart3d, p)
    {'r': Array(2., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, ...)}

    """
    del to_chart, from_chart  # Unused
    x, y, z = p["x"], p["y"], p["z"]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r == 0, jnp.ones(r.shape), z / r))
    # atan2 handles the case when x = y = 0, returning phi = 0
    phi = jnp.atan2(y, x)
    return {"r": r, "theta": theta, "phi": phi}


@plum.dispatch
def point_transform(
    to_chart: cxc.Spherical3D,
    from_chart: cxc.Cylindrical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Cylindrical3D -> Spherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(1.0, "m")}
    >>> cxt.point_transform(cxc.sph3d, cxc.cyl3d, p)
    {'r': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(0., dtype=float64), unit='rad'),
     'phi': Quantity(Array(0, dtype=int64, ...), unit='rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": 3.0, "phi": 0, "z": 0.0}  # No units
    >>> cxt.point_transform(cxc.sph3d, cxc.cyl3d, p)
    {'r': Array(3., dtype=float64, ...), 'theta': Array(1.57079633, dtype=float64),
     'phi': 0}

    """
    del to_chart, from_chart  # Unused
    r_ = jnp.hypot(p["rho"], p["z"])
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    return {"r": r_, "theta": theta, "phi": p["phi"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.Cylindrical3D,
    from_chart: cxc.Spherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Spherical3D -> Cylindrical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxt.point_transform(cxc.cyl3d, cxc.sph3d, p)
    {'rho': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'phi': Quantity(Array(0, dtype=int64, ...), unit='rad'),
     'z': Quantity(Array(1., dtype=float64, ...), unit='m')}

    A point on the equator (theta=90 deg):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.cyl3d, cxc.sph3d, p, usys=usys)
    {'rho': Array(2., dtype=float64, ...), 'phi': 0,
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    del to_chart, from_chart  # Unused
    theta = uconvert_to_rad(p["theta"], usys)
    rho = p["r"] * jnp.sin(theta)
    z = p["r"] * jnp.cos(theta)
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.LonLatSpherical3D,
    from_chart: cxc.Spherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Spherical3D -> LonLatSpherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Spherical theta=0 corresponds to lat=90 (north pole):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxt.point_transform(cxc.lonlatsph3d, cxc.sph3d, p)
    {'lon': Quantity(Array(0, dtype=int64, ...), unit='rad'),
     'lat': Quantity(Array(90., dtype=float64), unit='deg'),
     'distance': Quantity(Array(1., dtype=float64, ...), unit='m')}

    Spherical theta=90 deg corresponds to lat=0 (equator):

    >>> p = {"r": 1.0, "theta": 0, "phi": 0}  # No units
    >>> cxt.point_transform(cxc.lonlatsph3d, cxc.sph3d, p)
    {'lon': 0, 'lat': 1.5707963267948966, 'distance': 1.0}

    """
    del to_chart, from_chart  # Unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    return {"lon": p["phi"], "lat": lat, "distance": p["r"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.LonCosLatSpherical3D,
    from_chart: cxc.Spherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Spherical3D -> LonCosLatSpherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    On the equator (theta=90 deg), lon_coslat equals lon:

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}
    >>> cxt.point_transform(cxc.loncoslatsph3d, cxc.sph3d, p)
    {'lon_coslat': Quantity(Array(45., dtype=float64, ...), unit='deg'),
     'lat': Quantity(Array(0., dtype=float64, ...), unit='deg'),
     'distance': Quantity(Array(1., dtype=float64, ...), unit='m')}

    At the north pole (theta=0), lon_coslat = 0 regardless of phi:

    >>> p = {"r": 1.0, "theta": 0, "phi": 45}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.loncoslatsph3d, cxc.sph3d, p, usys=usys)
    {'lon_coslat': Array(2.7554553e-15, dtype=float64, ...),
     'lat': 1.5707963267948966, 'distance': 1.0}

    """
    del to_chart, from_chart  # Unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat, "distance": p["r"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.MathSpherical3D,
    from_chart: cxc.Spherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Spherical3D -> MathSpherical3D.

    Swaps theta and phi: Physics (theta=polar, phi=azimuth) to
    Math (theta=azimuth, phi=polar).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxt.point_transform(cxc.mathsph3d, cxc.sph3d, p)
    {'r': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(60, dtype=int64, ...), unit='deg'),
     'phi': Quantity(Array(30, dtype=int64, ...), unit='deg')}

    >>> p = {"r": 1.0, "theta": 30, "phi": 60}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.mathsph3d, cxc.sph3d, p, usys=usys)
    {'r': 1.0, 'theta': 60, 'phi': 30}

    """
    del to_chart, from_chart, usys  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.Spherical3D,
    from_chart: cxc.MathSpherical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """MathSpherical3D -> Spherical3D.

    Swaps theta and phi: Math (theta=azimuth, phi=polar) to
    Physics (theta=polar, phi=azimuth).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxt.point_transform(cxc.sph3d, cxc.mathsph3d, p)
    {'r': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(30, dtype=int64, ...), unit='deg'),
     'phi': Quantity(Array(60, dtype=int64, ...), unit='deg')}

    >>> p = {"r": 1.0, "theta": 60, "phi": 30}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxt.point_transform(cxc.sph3d, cxc.mathsph3d, p, usys=usys)
    {'r': 1.0, 'theta': 30, 'phi': 60}

    """
    del to_chart, from_chart, usys  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


def point_transform(
    to_chart: cxc.Cylindrical3D,
    from_chart: cxc.ProlateSpheroidal3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    r"""ProlateSpheroidal3D -> Cylindrical3D.

    Uses the focal length $\Delta$ stored on ``from_chart``.

    Validity constraints (enforced by the representation) are:

    - $\Delta > 0$,
    - $\mu \ge \Delta^2$,
    - $|\nu| \le \Delta^2$.

    The conversion proceeds via

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$,
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\,\mathrm{sign}(\nu)$,
    $\phi = \phi$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxt.point_transform(cxc.cyl3d, prolatesph3d, p)
    {'rho': Quantity(Array(0.8660254, dtype=float64), unit='m'),
     'phi': Quantity(Array(0., dtype=float64), unit='rad'),
     'z': Quantity(Array(1.11803399, dtype=float64), unit='m')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxt.point_transform(cxc.cyl3d, prolatesph3d, p, usys=usys)
    {'rho': Array(0.8660254, dtype=float64), 'phi': Array(0., dtype=float64),
     'z': Array(1.11803399, dtype=float64)}

    """
    del to_chart  # Unused
    nu, mu = p["nu"], p["mu"]
    if not isinstance(nu, ABCQ) or not isinstance(mu, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'mu' or 'nu', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = u.ustrip(usys["area"], from_chart.Delta**2)
    else:
        Delta2 = from_chart.Delta**2
    nu_D2 = jnp.abs(nu) / Delta2
    rho = jnp.sqrt((mu - Delta2) * (1 - nu_D2))
    z = jnp.sqrt(mu * nu_D2) * jnp.sign(nu)
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def point_transform(
    to_chart: cxc.ProlateSpheroidal3D,
    from_chart: cxc.Cylindrical3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    r"""Cylindrical3D -> ProlateSpheroidal3D.

    Uses the focal length $\Delta$ stored on ``to_chart``.

    Let $R^2 = \rho^2$ and $z^2 = z^2$ and define

    $S = R^2 + z^2 + \Delta^2$,
    $D_f = R^2 + z^2 - \Delta^2$,
    $D = \sqrt{D_f^2 + 4 R^2 \Delta^2}$.

    Then

    $\mu = \Delta^2 + \tfrac12(D + D_f)$ (with numerically-stable branches),
    $|\nu| = \dfrac{2\Delta^2}{S + D}\,z^2$,
    and $\nu = |\nu|\,\mathrm{sign}(z)$, with a stability fix when
    $\Delta^2 - |\nu|$ is small.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(3.0, "m")}
    >>> cxt.point_transform(prolatesph3d, cxc.cyl3d, p)
    {'mu': Quantity(Array(9., dtype=float64), unit='m2'),
     'nu': Quantity(Array(4., dtype=float64), unit='m2'),
     'phi': Quantity(Array(0, dtype=int64, ...), unit='rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": u.Q(2.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(0.0, "m")}
    >>> cxt.point_transform(prolatesph3d, cxc.cyl3d, p)
    {'mu': Quantity(Array(8., dtype=float64), unit='m2'),
     'nu': Quantity(Array(0., dtype=float64), unit='m2'),
     'phi': Quantity(Array(0, dtype=int64, ...), unit='rad')}

    Without units:

    >>> p = {"rho": 2.0, "phi": 0, "z": 3.0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxt.point_transform(prolatesph3d, cxc.cyl3d, p, usys=usys)
    {'mu': Array(14.52079729, dtype=float64),
     'nu': Array(2.47920271, dtype=float64), 'phi': 0}

    """
    del from_chart  # Unused
    # Pre-compute common terms
    R2 = p["rho"] ** 2
    z2 = p["z"] ** 2
    if not isinstance(R2, ABCQ) or not isinstance(z2, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'rho' or 'z', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = u.ustrip(usys["length"], to_chart.Delta) ** 2
    else:
        # TODO: fix Delta**2 issue in unxt
        Delta2 = u.Q.from_(to_chart.Delta) ** 2

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

    return {"mu": mu, "nu": nu, "phi": p["phi"]}


@plum.dispatch
def point_transform(
    to_chart: cxc.ProlateSpheroidal3D,
    from_chart: cxc.ProlateSpheroidal3D,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    r"""ProlateSpheroidal3D -> ProlateSpheroidal3D.

    If the focal length is unchanged (``to_chart.Delta == from_chart.Delta``), this
    is the identity map.

    If the focal length changes, we convert via cylindrical coordinates:

    ``Prolate(Delta_in) -> Cylindrical -> Prolate(Delta_out)``.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Same focal length (identity transformation):

    >>> prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxt.point_transform(prolate, prolate, p)
    {'mu': Quantity(Array(5., dtype=float64), unit='m2'),
     'nu': Quantity(Array(1., dtype=float64), unit='m2'),
     'phi': Quantity(Array(0., dtype=float64), unit='rad')}

    Different focal lengths (converts via cylindrical):

    >>> prolate_in = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> prolate_out = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(3.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxt.point_transform(prolate_out, prolate_in, p)
    {'mu': Quantity(Array(9.85889894, dtype=float64), unit='m2'),
     'nu': Quantity(Array(1.14110106, dtype=float64), unit='m2'),
     'phi': Quantity(Array(0., dtype=float64), unit='rad')}

    """
    # Cast to the result type
    dtype = jnp.result_type(
        to_chart.Delta, from_chart.Delta, *[v.dtype for v in p.values()]
    )
    p = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), p)
    return jax.lax.cond(
        to_chart.Delta == from_chart.Delta,
        lambda p: p,
        lambda p: api.point_transform(
            to_chart,
            cxc.cyl3d,
            api.point_transform(cxc.cyl3d, from_chart, p, usys=usys),
            usys=usys,
        ),
        p,
    )


# -----------------------------------------------
# 4D


@plum.dispatch
def point_transform(
    to_chart: cxc.SpaceTimeEuclidean,  # type: ignore[type-arg]
    from_chart: cxc.SpaceTimeEuclidean,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """SpaceTimeEuclidean[SpatialKind1] -> SpaceTimeEuclidean[SpatialKind2].

    Transforms the spatial components while preserving the time coordinate.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Transform from Cartesian to Spherical spatial coordinates:

    >>> eucl_cart = cxc.SpaceTimeEuclidean(cxc.cart3d)
    >>> eucl_sph = cxc.SpaceTimeEuclidean(cxc.sph3d)
    >>> p = {"ct": u.Q(1.0, "s"), "x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"),
    ...      "z": u.Q(0.0, "m")}
    >>> cxt.point_transform(eucl_sph, eucl_cart, p)
    {'ct': Quantity(Array(1., dtype=float64, ...), unit='s'),
     'r': Quantity(Array(1., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    Transform between cylindrical and spherical:

    >>> eucl_cyl = cxc.SpaceTimeEuclidean(cxc.cyl3d)
    >>> p = {"ct": u.Q(2.0, "s"), "rho": u.Q(3.0, "m"), "phi": u.Q(0, "rad"),
    ...      "z": u.Q(4.0, "m")}
    >>> cxt.point_transform(eucl_sph, eucl_cyl, p)
    {'ct': Quantity(Array(2., dtype=float64, ...), unit='s'),
     'r': Quantity(Array(5., dtype=float64, ...), unit='m'),
     'theta': Quantity(Array(0.64350111, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0, dtype=int64, ...), unit='rad')}

    """
    return {
        "ct": p["ct"],
        **api.point_transform(
            to_chart.spatial_chart, from_chart.spatial_chart, p, usys=usys
        ),
    }


# -----------------------------------------------
# N-D


def _cartnd_to_cartesian(
    p: CsDict,
    cart_chart: cxc.AbstractChart,  # type: ignore[type-arg]
) -> CsDict:
    """Convert CartND data dict to a fixed-dimensional Cartesian dict.

    Extracts components from the 'q' array and maps them to named components.
    """
    q = p["q"]
    return {comp: q[i] for i, comp in enumerate(cart_chart.components)}


@plum.dispatch
def point_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.CartND,
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """CartND -> AbstractChart.

    Converts from N-dimensional Cartesian (with a single 'q' array) to any
    other chart type by first extracting the appropriate fixed-dimensional
    Cartesian representation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Convert 3D CartND to Spherical:

    >>> p = {"q": u.Q([1.0, 0.0, 0.0], "m")}
    >>> cxt.point_transform(cxc.sph3d, cxc.cartnd, p)
    {'r': Quantity(Array(1., dtype=float64), unit='m'),
     'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64), unit='rad')}

    Convert 2D CartND to Polar:

    >>> p = {"q": u.Q([3.0, 4.0], "m")}
    >>> cxt.point_transform(cxc.polar2d, cxc.cartnd, p)
    {'r': Quantity(Array(5., dtype=float64), unit='m'),
     'theta': Quantity(Array(0.92729522, dtype=float64), unit='rad')}

    Convert 1D CartND to Radial:

    >>> p = {"q": u.Q([5.0], "m")}
    >>> cxt.point_transform(cxc.radial1d, cxc.cartnd, p)
    {'r': Quantity(Array(5., dtype=float64), unit='m')}

    Convert CartND to Cart3D:

    >>> p = {"q": u.Q([1.0, 2.0, 3.0], "m")}
    >>> cxt.point_transform(cxc.cart3d, cxc.cartnd, p)
    {'x': Quantity(Array(1., dtype=float64), unit='m'),
     'y': Quantity(Array(2., dtype=float64), unit='m'),
     'z': Quantity(Array(3., dtype=float64), unit='m')}

    """
    del from_chart  # Unused

    # If target is CartND, we can't convert (would be infinite recursion)
    if isinstance(to_chart, cxc.CartND):
        msg = "Cannot convert CartND to CartND via this dispatch."
        raise TypeError(msg)

    # Get the corresponding fixed-dimensional Cartesian chart
    cart_chart = to_chart.cartesian

    # If cartesian_chart returns CartND, we don't support this conversion
    if isinstance(cart_chart, cxc.CartND):
        msg = f"CartND conversion not supported for {type(to_chart).__name__}."
        raise NotImplementedError(msg)

    # Get dimensionality from the target chart
    target_ndim = to_chart.ndim

    # Check that the CartND data has the right dimensionality
    q = p["q"]
    data_ndim = q.shape[0]
    eqx.error_if(
        q,
        data_ndim != target_ndim,
        f"CartND data has {data_ndim} dimensions but target chart "
        f"{type(to_chart).__name__} requires {target_ndim} dimensions.",
    )

    # Convert CartND to fixed-dimensional Cartesian
    p_cart = _cartnd_to_cartesian(p, cart_chart)

    # If target is already the Cartesian chart, return directly
    if type(to_chart) is type(cart_chart):
        return p_cart

    # Otherwise, transform from Cartesian to target chart
    return api.point_transform(to_chart, cart_chart, p_cart, usys=usys)


def _cartesian_to_cartnd(
    p: CsDict,
    cart_chart: cxc.AbstractChart,  # type: ignore[type-arg]
) -> CsDict:
    """Convert a fixed-dimensional Cartesian dict to CartND data dict.

    Stacks named components into a single 'q' array.
    """
    q = jnp.stack([p[comp] for comp in cart_chart.components], axis=0)
    return {"q": q}


@plum.dispatch
def point_transform(
    to_chart: cxc.CartND,
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """AbstractChart -> CartND.

    Converts from any chart type to N-dimensional Cartesian (with a single
    'q' array) by first transforming to the appropriate fixed-dimensional
    Cartesian representation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Convert Cart3D to CartND:

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> cxt.point_transform(cxc.cartnd, cxc.cart3d, p)
    {'q': Quantity(Array([1., 2., 3.], dtype=float64, ...), unit='m')}

    Convert Cart2D to CartND:

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> cxt.point_transform(cxc.cartnd, cxc.cart2d, p)
    {'q': Quantity(Array([3., 4.], dtype=float64, ...), unit='m')}

    Convert Radial to CartND:

    >>> p = {"r": u.Q(3.0, "m")}
    >>> cxt.point_transform(cxc.cartnd, cxc.radial1d, p)
    {'q': Quantity(Array([3.], dtype=float64, ...), unit='m')}

    Convert Cylindrical to CartND (z-axis point):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(5.0, "m")}
    >>> cxt.point_transform(cxc.cartnd, cxc.cyl3d, p)
    {'q': Quantity(Array([0., 0., 5.], dtype=float64, ...), unit='m')}

    """
    del to_chart  # Unused

    # If source is CartND, we can't convert (would be infinite recursion)
    if isinstance(from_chart, cxc.CartND):
        msg = "Cannot convert CartND to CartND via this dispatch."
        raise TypeError(msg)

    # Get the corresponding fixed-dimensional Cartesian chart
    cart_chart = from_chart.cartesian

    # If cartesian_chart returns CartND, we don't support this conversion
    if isinstance(cart_chart, cxc.CartND):
        msg = f"CartND conversion not supported for {type(from_chart).__name__}."
        raise NotImplementedError(msg)

    # Transform from source to fixed-dimensional Cartesian
    p_cart = api.point_transform(cart_chart, from_chart, p, usys=usys)

    # Convert fixed-dimensional Cartesian to CartND
    return _cartesian_to_cartnd(p_cart, cart_chart)


# ===================================================================
# Embedded manifold conversions


@plum.dispatch
def point_transform(
    to_chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    from_chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Convert between embedded manifolds with a shared ambient space.

    This function transforms intrinsic coordinates from one embedded manifold
    to another by:
    1. Embedding the point into the ambient space of the source manifold
    2. Transforming in the ambient space (if the ambient charts differ)
    3. Projecting back to the intrinsic coordinates of the target manifold

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **Example 1: Two spheres with different radii**

    Both spheres use the same intrinsic TwoSphere chart but have different
    radii as parameters:

    >>> sphere1 = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                                params={"R": u.Q(1.0, "km")})
    >>> sphere2 = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                                params={"R": u.Q(2.0, "km")})

    A point on sphere1 (theta=pi/4, phi=0):

    >>> p = {"theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> p2 = cxt.point_transform(sphere2, sphere1, p)
    >>> {k: v.uconvert("deg") for k, v in p2.items()}
    {'theta': Quantity(Array(45., dtype=float64, ...), unit='deg'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='deg')}

    The angular coordinates are preserved (both spheres share the same
    angular parameterization via projection through the shared ambient space).

    """
    if type(to_chart.ambient_chart) is not type(from_chart.ambient_chart):
        msg = "EmbeddedManifold ambient kinds must match for conversion."
        raise ValueError(msg)

    p_ambient = api.embed_point(from_chart, p)  # TODO: support usys
    p_ambient = api.point_transform(
        to_chart.ambient_chart, from_chart.ambient_chart, p_ambient, usys=usys
    )
    return api.project_point(to_chart, p_ambient)  # TODO: support usys


@plum.dispatch
def point_transform(
    to_chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Project an ambient position into an embedded chart.

    This transforms coordinates from an ambient chart (e.g., Cartesian or
    Spherical) into the intrinsic coordinates of an embedded manifold.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **From Cartesian ambient to TwoSphere intrinsic:**

    >>> sphere = cxc.EmbeddedManifold(cxc.twosphere,cxc.cart3d,
    ...                               params={"R": u.Q(1.0, "m")})

    A point on the unit sphere in Cartesian coords (on equator, x-axis):

    >>> p_cart = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxt.point_transform(sphere, cxc.cart3d, p_cart)
    {'theta': Quantity(Array(1.57079633, dtype=float64, ...), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    **From Spherical ambient to TwoSphere intrinsic:**

    The ambient spherical coords (r, theta, phi) project to intrinsic
    (theta, phi), discarding the radial component:

    >>> p_sph = {"r": 5, "theta": 1, "phi": 0.5}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxt.point_transform(sphere, cxc.sph3d, p_sph, usys=usys)
    {'theta': Array(1., dtype=float64, ...),
     'phi': Array(0.5, dtype=float64, ...)}

    """
    p_ambient = api.point_transform(to_chart.ambient_chart, from_chart, p, usys=usys)
    return api.project_point(to_chart, p_ambient)


@plum.dispatch
def point_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """Embed intrinsic coordinates into an ambient representation.

    This transforms intrinsic coordinates of an embedded manifold into
    coordinates of an ambient chart, which may differ from the embedding's
    native ambient chart.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **From TwoSphere intrinsic to Cartesian ambient:**

    >>> sphere = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                               params={"R": u.Q(1.0, "m")})

    A point on the unit sphere in (on equator, x-axis):

    >>> p_cart = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.0, "rad")}
    >>> cxt.point_transform(cxc.cart3d, sphere, p_cart)
    {'x': Quantity(Array(0.84147098, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(0.54030231, dtype=float64, ...), unit='m')}

    **From TwoSphere intrinsic to Spherical ambient:**

    >>> p_sph = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxt.point_transform(cxc.sph3d, sphere, p_sph)
    {'r': Quantity(Array(1., dtype=float64, weak_type=True), unit='m'),
     'theta': Quantity(Array(1., dtype=float64), unit='rad'),
     'phi': Quantity(Array(0.5, dtype=float64, weak_type=True), unit='rad')}

    """
    p_ambient = api.embed_point(from_chart, p)
    return api.point_transform(to_chart, from_chart.ambient_chart, p_ambient, usys=usys)


# ===================================================================
# Cartesian Product Chart conversions


@plum.dispatch
def point_transform(
    to_chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """AbstractCartesianProductChart -> AbstractCartesianProductChart (factorwise).

    Transforms between product charts by applying point_transform to each factor
    independently. Requires compatible factor structure (same number of factors,
    pairwise compatible).

    Mathematical definition:
        point_transform(∏ᵢ Sᵢ, ∏ᵢ Rᵢ, p) = (point_transform(Sᵢ, Rᵢ, pᵢ))ᵢ

    where pᵢ are the factor dictionaries split from p.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Transform SpaceTimeCT (a product chart) between spatial representations:

    >>> spacetime_cart = cxc.SpaceTimeCT(cxc.cart3d)
    >>> spacetime_sph = cxc.SpaceTimeCT(cxc.sph3d)
    >>> p = {"ct": u.Q(1.0, "s"), "x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"),
    ...      "z": u.Q(0.0, "m")}
    >>> result = cxt.point_transform(spacetime_sph, spacetime_cart, p)
    >>> result["ct"]
    Quantity(Array(1., dtype=float64, ...), unit='s')
    >>> result["r"]
    Quantity(Array(1., dtype=float64, ...), unit='m')

    """
    # Product charts can't safely use the cartesian intermediate because their
    # cartesian version is still a product chart, which would recurse here. Do
    # a factor-wise transform directly instead.
    if len(to_chart.factors) != len(from_chart.factors):
        msg = (
            "Cannot transform between product charts with different numbers of "
            "factors: "
            f"{len(from_chart.factors)} -> {len(to_chart.factors)}"
        )
        raise TypeError(msg)

    parts = from_chart.split_components(p)
    transformed = tuple(
        api.point_transform(t_factor, f_factor, p_part, usys=usys)
        for t_factor, f_factor, p_part in zip(
            to_chart.factors, from_chart.factors, parts, strict=True
        )
    )
    return to_chart.merge_components(transformed)


@plum.dispatch
def point_transform(
    to_chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """AbstractChart -> Cartesian -> AbstractChart."""
    msg = (
        f"No general transform between {type(from_chart).__name__} and "
        f"{type(to_chart).__name__}. Define explicit rules for non-product to "
        "product conversions."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def point_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    """AbstractChart -> Cartesian -> AbstractChart."""
    msg = (
        f"No general transform between {type(from_chart).__name__} and "
        f"{type(to_chart).__name__}. Define explicit rules for product to "
        "non-product conversions."
    )
    raise NotImplementedError(msg)


# ===================================================================
# Point Transform an Array


@plum.dispatch
def point_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,
    p: ArrayLike,
    /,
    usys: OptUSys,
) -> Array:
    r"""Point transform for array input."""
    # Build a dict of arrays for each component
    p_dict = api.cdict(p, from_chart)

    # Transform the point dict
    p_transformed = api.point_transform(to_chart, from_chart, p_dict, usys=usys)

    # Stack the transformed components into an array
    p_out = jnp.stack([p_transformed[comp] for comp in to_chart.components], axis=-1)

    return p_out
