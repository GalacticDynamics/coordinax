"""Conversion functions for vector representations."""

from collections.abc import Mapping

import jax
import plum

import quaxed.numpy as jnp
import unxt as u

from . import api, euclidean as r
from .custom_types import PDict
from .embed import EmbeddedManifold, embed_pos, project_pos
from .manifolds import TwoSphere
from .roles import Pos
from .spacetime import SpaceTimeCT


@plum.dispatch
def vconvert(
    role: Pos,
    to_rep: r.AbstractRep,
    from_rep: r.AbstractRep,
    p: PDict,
    /,
    *_: PDict,
) -> PDict:
    """Convert a position vector from one representation to another."""
    # Convert using the representation conversion
    return api.coord_map(to_rep, from_rep, p)


# ===================================================================
# Embedded manifold conversions


@plum.dispatch
def coord_map(
    to_rep: EmbeddedManifold, from_rep: EmbeddedManifold, p: PDict, /
) -> PDict:
    """Convert between embedded manifolds with a shared ambient space."""
    if type(to_rep.ambient_kind) is not type(from_rep.ambient_kind):
        msg = "EmbeddedManifold ambient kinds must match for conversion."
        raise ValueError(msg)

    p_ambient = embed_pos(from_rep, p)
    p_ambient = api.coord_map(to_rep.ambient_kind, from_rep.ambient_kind, p_ambient)
    return project_pos(to_rep, p_ambient)


@plum.dispatch
def coord_map(to_rep: EmbeddedManifold, from_rep: r.AbstractRep, p: PDict, /) -> PDict:
    """Project an ambient position into an embedded chart."""
    p_ambient = api.coord_map(to_rep.ambient_kind, from_rep, p)
    return project_pos(to_rep, p_ambient)


@plum.dispatch
def coord_map(to_rep: r.AbstractRep, from_rep: EmbeddedManifold, p: PDict, /) -> PDict:
    """Embed intrinsic coordinates into an ambient representation."""
    p_ambient = embed_pos(from_rep, p)
    return api.coord_map(to_rep, from_rep.ambient_kind, p_ambient)


# ===================================================================
# Self representation conversions


@plum.dispatch.multi(
    *(
        (typ, typ, PDict)
        for typ in [
            # 0D
            r.Cart0D,
            # 1D
            r.Cart1D,
            r.Radial1D,
            # 2D
            r.Cart2D,
            r.Polar2D,
            TwoSphere,
            # 3D
            r.Cart3D,
            r.Cylindrical3D,
            r.Spherical3D,
            r.LonLatSpherical3D,
            r.LonCosLatSpherical3D,
            r.MathSpherical3D,
            # r.ProlateSpheroidal3D,  # requires Delta
            # r.ProlateSpheroidalVel,  # requires Delta
            # r.ProlateSpheroidalAcc,  # requires Delta
            # 4D
            # SpaceTimeCT,  # depends on spatial representation
            # 6D
            r.PoincarePolar6D,
            # N-D
            r.CartND,
        ]
    )
)
def coord_map(
    to_rep: r.AbstractRep,
    from_rep: r.AbstractRep,
    p: PDict,
    /,
) -> PDict:
    return p


# ---------------------------------------------------------
# General representation conversions


@plum.dispatch(precedence=-1)
def coord_map(to_rep: r.AbstractRep, from_rep: r.AbstractRep, p: PDict, /) -> PDict:
    """AbstractRep -> Cartesian -> AbstractRep."""
    p_cart = api.coord_map(to_rep.cartesian, from_rep, p)
    return api.coord_map(to_rep, from_rep.cartesian, p_cart)


# ---------------------------------------------------------
# Specific representation conversions

# -----------------------------------------------
# 1D


@plum.dispatch
def coord_map(to_rep: r.Cart1D, from_rep: r.Radial1D, p: PDict, /) -> PDict:
    """Radial1D -> Cartesian1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return {"x": p["r"]}


@plum.dispatch
def coord_map(to_rep: r.Radial1D, from_rep: r.Cart1D, p: PDict, /) -> PDict:
    """Cartesian1D -> Radial1D.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.
    """
    del to_rep, from_rep  # Unused
    return {"r": p["x"]}


# -----------------------------------------------
# 2D


@plum.dispatch
def coord_map(to_rep: r.Cart2D, from_rep: r.Polar2D, p: PDict, /) -> PDict:
    """Polar2D -> Cart2D.

    The `r` and `theta` coordinates are converted to the `x` and `y` coordinates
    of the 2D Cartesian system.
    """
    del to_rep, from_rep  # Unused
    x = p["r"] * jnp.cos(p["theta"])
    y = p["r"] * jnp.sin(p["theta"])
    return {"x": x, "y": y}


@plum.dispatch
def coord_map(to_rep: r.Polar2D, from_rep: r.Cart2D, p: PDict, /) -> PDict:
    """Cart2D -> Polar2D.

    The `x` and `y` coordinates are converted to the `r` and `theta` coordinates
    of the 2D polar system.
    """
    del to_rep, from_rep  # Unused
    r_ = jnp.hypot(p["x"], p["y"])
    theta = jnp.arctan2(p["y"], p["x"])
    return {"r": r_, "theta": theta}


# -----------------------------------------------
# 3D


@plum.dispatch
def coord_map(to_rep: r.Cart3D, from_rep: r.Cylindrical3D, p: PDict, /) -> PDict:
    """Cylindrical3D -> Cart3D."""
    del to_rep, from_rep  # Unused
    x = p["rho"] * jnp.cos(p["phi"])
    y = p["rho"] * jnp.sin(p["phi"])
    return {"x": x, "y": y, "z": p["z"]}


@plum.dispatch
def coord_map(to_rep: r.Cart3D, from_rep: r.Spherical3D, p: PDict, /) -> PDict:
    """Spherical3D -> Cart3D."""
    del to_rep, from_rep  # Unused
    r, theta, phi = p["r"], p["theta"], p["phi"]
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def coord_map(to_rep: r.Cart3D, from_rep: r.LonLatSpherical3D, p: PDict, /) -> PDict:
    """LonLatSpherical3D -> Cart3D."""
    del to_rep, from_rep
    lon, lat, r = p["lon"], p["lat"], p["distance"]
    x = r * jnp.cos(lat) * jnp.cos(lon)
    y = r * jnp.cos(lat) * jnp.sin(lon)
    z = r * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def coord_map(to_rep: r.Cart3D, from_rep: r.LonCosLatSpherical3D, p: PDict, /) -> PDict:
    """LonCosLatSpherical3D -> Cart3D.

    Components are (lon_coslat, lat, r), where lon_coslat := lon * cos(lat).
    Longitude is undefined at the poles (cos(lat) == 0); we set lon = 0 by
    convention there to avoid NaNs.
    """
    del to_rep, from_rep
    lon_coslat, lat, r = p["lon_coslat"], p["lat"], p["distance"]
    # Handle the poles where cos(lat) == 0
    coslat = jnp.cos(lat)
    lon = jnp.where(coslat == 0, 0, lon_coslat / coslat)
    # Convert to Cartesian
    x = r * jnp.cos(lat) * jnp.cos(lon)
    y = r * jnp.cos(lat) * jnp.sin(lon)
    z = r * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def coord_map(to_rep: r.Cart3D, from_rep: r.MathSpherical3D, p: PDict, /) -> PDict:
    """MathSpherical3D -> Cart3D.

    - theta: azimuth in the x-y plane (longitude-like)
    - phi  : polar angle from +z, with phi in [0, pi]

    """
    del to_rep, from_rep  # Unused
    r_, theta, phi = p["r"], p["theta"], p["phi"]
    x = r_ * jnp.sin(phi) * jnp.cos(theta)
    y = r_ * jnp.sin(phi) * jnp.sin(theta)
    z = r_ * jnp.cos(phi)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def coord_map(to_rep: r.Cart3D, from_rep: r.ProlateSpheroidal3D, p: PDict, /) -> PDict:
    r"""ProlateSpheroidal3D -> Cart3D.

    We calculate through cylindrical coordinates first:

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\;\mathrm{sign}(\nu)$
    $\phi = \phi.$

    Then convert to Cartesian:

    $x=\rho\cos\phi$, $y=\rho\sin\phi$, $z=z$.

    """
    # Calculate cylindrical distance
    Delta2 = from_rep.Delta**2
    nu_D2 = jnp.abs(p["nu"]) / Delta2
    rho = jnp.sqrt((p["mu"] - Delta2) * (1 - nu_D2))
    # Convert to Cartesian
    x = rho * jnp.cos(p["phi"])
    y = rho * jnp.sin(p["phi"])
    z = jnp.sqrt(p["mu"] * nu_D2) * jnp.sign(p["nu"])
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def coord_map(to_rep: r.Cylindrical3D, from_rep: r.Cart3D, p: PDict, /) -> PDict:
    """Cart3D -> Cylindrical3D."""
    del to_rep, from_rep  # Unused
    rho = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"rho": rho, "phi": phi, "z": p["z"]}


@plum.dispatch.multi(
    (r.AbstractSpherical3D, r.Cart3D, Mapping),
    (r.AbstractSpherical3D, r.Cylindrical3D, Mapping),
)
def coord_map(
    to_rep: r.AbstractSpherical3D,
    from_rep: r.Cart3D | r.Cylindrical3D,
    p: PDict,
    /,
) -> PDict:
    """Cart3D -> Spherical3D -> AbstractSpherical3D."""
    p_sph = api.coord_map(r.sph3d, from_rep, p)
    return api.coord_map(to_rep, r.sph3d, p_sph)


@plum.dispatch
def coord_map(to_rep: r.Spherical3D, from_rep: r.Cart3D, p: PDict, /) -> PDict:
    """Cart3D -> Spherical3D."""
    del to_rep, from_rep  # Unused
    r_ = jnp.sqrt(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2)
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    # atan2 handles the case when x = y = 0, returning phi = 0
    phi = jnp.atan2(p["y"], p["x"])
    return {"r": r_, "theta": theta, "phi": phi}


@plum.dispatch
def coord_map(to_rep: r.Spherical3D, from_rep: r.Cylindrical3D, p: PDict, /) -> PDict:
    """Cylindrical3D -> Spherical3D."""
    del to_rep, from_rep  # Unused
    r_ = jnp.hypot(p["rho"], p["z"])
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    return {"r": r_, "theta": theta, "phi": p["phi"]}


@plum.dispatch
def coord_map(to_rep: r.Cylindrical3D, from_rep: r.Spherical3D, p: PDict, /) -> PDict:
    """Spherical3D -> Cylindrical3D."""
    del to_rep, from_rep  # Unused
    rho = p["r"] * jnp.sin(p["theta"])
    z = p["r"] * jnp.cos(p["theta"])
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def coord_map(
    to_rep: r.LonLatSpherical3D, from_rep: r.Spherical3D, p: PDict, /
) -> PDict:
    """Spherical3D -> LonLatSpherical3D."""
    del to_rep, from_rep  # Unused
    lat = (
        u.Quantity(90, "deg")
        if isinstance(p["theta"], u.AbstractQuantity)
        else jnp.pi / 2
    ) - p["theta"]
    return {"lon": p["phi"], "lat": lat, "distance": p["r"]}


@plum.dispatch
def coord_map(
    to_rep: r.LonCosLatSpherical3D, from_rep: r.Spherical3D, p: PDict, /
) -> PDict:
    """Spherical3D -> LonCosLatSpherical3D."""
    del to_rep, from_rep  # Unused
    lat = u.Quantity(90, "deg") - p["theta"]
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat, "distance": p["r"]}


@plum.dispatch
def coord_map(to_rep: r.MathSpherical3D, from_rep: r.Spherical3D, p: PDict, /) -> PDict:
    del to_rep, from_rep  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def coord_map(to_rep: r.Spherical3D, from_rep: r.MathSpherical3D, p: PDict, /) -> PDict:
    del to_rep, from_rep  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


def coord_map(
    to_rep: r.Cylindrical3D, from_rep: r.ProlateSpheroidal3D, p: PDict, /
) -> PDict:
    r"""ProlateSpheroidal3D -> Cylindrical3D.

    Uses the focal length $\Delta$ stored on ``from_rep``.

    Validity constraints (enforced by the representation) are:

    - $\Delta > 0$,
    - $\mu \ge \Delta^2$,
    - $|\nu| \le \Delta^2$.

    The conversion proceeds via

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$,
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\,\mathrm{sign}(\nu)$,
    $\phi = \phi$.

    """
    del to_rep  # Unused
    Delta2 = from_rep.Delta**2
    nu_D2 = jnp.abs(p["nu"]) / Delta2
    rho = jnp.sqrt((p["mu"] - Delta2) * (1 - nu_D2))
    z = jnp.sqrt(p["mu"] * nu_D2) * jnp.sign(p["nu"])
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def coord_map(
    to_rep: r.ProlateSpheroidal3D, from_rep: r.Cylindrical3D, p: PDict, /
) -> PDict:
    r"""Cylindrical3D -> ProlateSpheroidal3D.

    Uses the focal length $\Delta$ stored on ``to_rep``.

    Let $R^2 = \rho^2$ and $z^2 = z^2$ and define

    $S = R^2 + z^2 + \Delta^2$,
    $D_f = R^2 + z^2 - \Delta^2$,
    $D = \sqrt{D_f^2 + 4 R^2 \Delta^2}$.

    Then

    $\mu = \Delta^2 + \tfrac12(D + D_f)$ (with numerically-stable branches),
    $|\nu| = \dfrac{2\Delta^2}{S + D}\,z^2$,
    and $\nu = |\nu|\,\mathrm{sign}(z)$, with a stability fix when
    $\Delta^2 - |\nu|$ is small.

    """
    del from_rep  # Unused
    # Pre-compute common terms
    R2 = p["rho"] ** 2
    z2 = p["z"] ** 2
    Delta2 = to_rep.Delta**2

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
def coord_map(
    to_rep: r.ProlateSpheroidal3D, from_rep: r.ProlateSpheroidal3D, p: PDict, /
) -> PDict:
    r"""ProlateSpheroidal3D -> ProlateSpheroidal3D.

    If the focal length is unchanged (``to_rep.Delta == from_rep.Delta``), this
    is the identity map.

    If the focal length changes, we convert via cylindrical coordinates:

    ``Prolate(Delta_in) -> Cylindrical -> Prolate(Delta_out)``.

    """
    # Cast to the result type
    dtype = jnp.result_type(
        to_rep.Delta, from_rep.Delta, *[v.dtype for v in p.values()]
    )
    p = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), p)
    return jax.lax.cond(
        to_rep.Delta == from_rep.Delta,
        lambda p: p,
        lambda p: api.coord_map(to_rep, r.cyl3d, api.coord_map(r.cyl3d, from_rep, p)),
        p,
    )


# -----------------------------------------------
# 4D


@plum.dispatch
def coord_map(to_rep: SpaceTimeCT, from_rep: SpaceTimeCT, p: PDict, /) -> PDict:
    """SpaceTimeCT[SpatialKind1] -> SpaceTimeCT[SpatialKind2]."""
    return {
        "ct": p["ct"],
        **api.coord_map(to_rep.spatial_kind, from_rep.spatial_kind, p),
    }


@plum.dispatch
def coord_map(
    to_rep: r.SpaceTimeEuclidean, from_rep: r.SpaceTimeEuclidean, p: PDict, /
) -> PDict:
    """SpaceTimeEuclidean[SpatialKind1] -> SpaceTimeEuclidean[SpatialKind2]."""
    return {
        "ct": p["ct"],
        **api.coord_map(to_rep.spatial_kind, from_rep.spatial_kind, p),
    }
