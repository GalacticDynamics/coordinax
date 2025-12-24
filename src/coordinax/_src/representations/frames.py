"""Orthonormal frame utilities for representations."""

__all__ = ("frame_to_cart", "pullback", "pushforward")

from jaxtyping import Array

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from . import euclidean as r
from .custom_types import PDict
from .embed import EmbeddedManifold
from .manifolds import TwoSphere
from .metrics import AbstractMetric, MinkowskiMetric
from .spacetime import SpaceTimeCT


@plum.dispatch
def frame_to_cart(rep: r.AbstractRep, p_pos: PDict, /) -> Array:
    r"""Return an orthonormal frame expressed in ambient Cartesian components.

    Mathematical definition
    -----------------------
    .. math::
       B(q) = [\hat e_1(q)\ \cdots\ \hat e_n(q)]
       \\
       \hat e_i \cdot \hat e_j = \delta_{ij} \quad \text{(Euclidean)}

    Parameters
    ----------
    rep
        Representation (rep) whose orthonormal frame is requested.
    p_pos
        Coordinate values keyed by ``rep.components``.

    Returns
    -------
    Array
        Matrix of shape ``(n_{\text{ambient}}, n_{\text{rep}})`` with columns
        equal to the orthonormal frame vectors expressed in ambient Cartesian
        components.

    Notes
    -----
    - For Euclidean 3D reps, ``n_ambient = n_rep = 3``.
    - For embedded manifolds, the frame is rectangular (e.g. ``3×2`` for ``S^2``).
    - For ``SpaceTimeCT``, orthonormality is with respect to the Minkowski metric
      with signature ``(-,+,+,+)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> p = {"r": u.Quantity(1.0, "km"), "theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.5, "rad")}
    >>> B = cx.r.frame_to_cart(cx.r.sph3d, p)
    >>> bool(jnp.allclose(B.T @ B, jnp.eye(3)))
    True

    """
    msg = f"No frame_to_cart rule registered for {type(rep)!r}."
    raise NotImplementedError(msg)


@plum.dispatch
def frame_to_cart(rep: EmbeddedManifold, p_pos: PDict, /) -> Array:
    """Return the orthonormal tangent frame in ambient Cartesian components."""
    if isinstance(rep.chart_kind, TwoSphere) and isinstance(rep.ambient_kind, r.Cart3D):
        theta = u.ustrip(AllowValue, "rad", p_pos["theta"])
        phi = u.ustrip(AllowValue, "rad", p_pos["phi"])

        e_theta = jnp.array(
            [
                jnp.cos(theta) * jnp.cos(phi),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        )
        e_phi = jnp.array(
            [
                -jnp.sin(phi),
                jnp.cos(phi),
                jnp.zeros_like(phi),
            ]
        )

        return jnp.stack([e_theta, e_phi], axis=-1)

    msg = f"No frame_to_cart rule registered for {type(rep.chart_kind)!r}."
    raise NotImplementedError(msg)


@plum.dispatch
def frame_to_cart(rep: r.Cart3D, p_pos: PDict, /) -> Array:  # type: ignore[attr-defined]
    del rep, p_pos
    return jnp.eye(3)


@plum.dispatch
def frame_to_cart(rep: r.Cylindrical3D, p_pos: PDict, /) -> Array:  # type: ignore[attr-defined]
    """Cylindrical orthonormal frame (e_rho, e_phi, e_z) in Cartesian components."""
    del rep
    phi = u.ustrip(AllowValue, "rad", p_pos["phi"])
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@plum.dispatch
def frame_to_cart(rep: r.Spherical3D, p_pos: PDict, /) -> Array:
    """Spherical (physics convention) orthonormal frame in Cartesian components."""
    del rep
    theta = u.ustrip(AllowValue, "rad", p_pos["theta"])
    phi = u.ustrip(AllowValue, "rad", p_pos["phi"])

    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)

    e_r = jnp.array([st * cp, st * sp, ct])
    e_th = jnp.array([ct * cp, ct * sp, -st])
    e_ph = jnp.array([-sp, cp, 0])
    return jnp.stack([e_r, e_th, e_ph], axis=-1)


@plum.dispatch
def frame_to_cart(rep: r.LonLatSpherical3D, p_pos: PDict, /) -> Array:
    """Lon/Lat spherical orthonormal frame in Cartesian components."""
    del rep
    lat = u.ustrip(AllowValue, "rad", p_pos["lat"])
    lon = u.ustrip(AllowValue, "rad", p_pos["lon"])
    theta = (jnp.pi / 2) - lat
    sph_pos = {"r": p_pos["distance"], "theta": theta, "phi": lon}
    return frame_to_cart(r.Spherical3D(), sph_pos)  # type: ignore[call-arg]


@plum.dispatch
def frame_to_cart(rep: r.MathSpherical3D, p_pos: PDict, /) -> Array:
    """MathSpherical3D orthonormal frame in Cartesian components."""
    del rep
    sph_pos = {"r": p_pos["r"], "theta": p_pos["phi"], "phi": p_pos["theta"]}
    return frame_to_cart(r.Spherical3D(), sph_pos)  # type: ignore[call-arg]


@plum.dispatch
def frame_to_cart(rep: r.LonCosLatSpherical3D, p_pos: PDict, /) -> Array:
    """Lon*cos(lat), lat, distance spherical orthonormal frame in Cartesian."""
    del rep
    lon_coslat = p_pos["lon_coslat"]
    lat = u.ustrip(AllowValue, "rad", p_pos["lat"])
    dist = p_pos["distance"]

    cos_lat = jnp.cos(lat)
    cos_lat_val = u.ustrip(AllowValue, cos_lat)

    eps = jnp.asarray(1e-12, dtype=cos_lat_val.dtype)
    safe = jnp.abs(cos_lat_val) > eps

    lon = lon_coslat / jnp.where(safe, cos_lat_val, 1.0)
    lon0 = u.Quantity(0, u.unit_of(lon_coslat))
    lon = jnp.where(safe, lon, lon0)

    theta = (jnp.pi / 2) - lat
    sph_pos = {"r": dist, "theta": theta, "phi": lon}
    return frame_to_cart(r.Spherical3D(), sph_pos)  # type: ignore[call-arg]


@plum.dispatch
def frame_to_cart(rep: SpaceTimeCT, p_pos: PDict, /) -> Array:
    """SpaceTimeCT orthonormal frame in Cartesian (ct,x,y,z) components."""
    sp = rep.spatial_kind
    p_sp = {k: p_pos[k] for k in sp.components}
    B_sp = frame_to_cart(sp, p_sp)

    n_sp = B_sp.shape[-1]
    one = jnp.ones((1, 1), dtype=B_sp.dtype)
    z1 = jnp.zeros((1, n_sp), dtype=B_sp.dtype)
    z3 = jnp.zeros((n_sp, 1), dtype=B_sp.dtype)
    return jnp.block([[one, z1], [z3, B_sp]])


@plum.dispatch
def frame_to_cart(rep: r.SpaceTimeEuclidean, p_pos: PDict, /) -> Array:
    """SpaceTimeEuclidean orthonormal frame in Cartesian (ct,x,y,z) components."""
    sp = rep.spatial_kind
    p_sp = {k: p_pos[k] for k in sp.components}
    B_sp = frame_to_cart(sp, p_sp)

    n_sp = B_sp.shape[-1]
    one = jnp.ones((1, 1), dtype=B_sp.dtype)
    z1 = jnp.zeros((1, n_sp), dtype=B_sp.dtype)
    z3 = jnp.zeros((n_sp, 1), dtype=B_sp.dtype)
    return jnp.block([[one, z1], [z3, B_sp]])


def pushforward(B: Array, v_rep: Array, /) -> Array:
    """Push forward components from a rep frame into Cartesian components."""
    return jnp.einsum("...ij,...j->...i", B, v_rep)


def pullback(metric: AbstractMetric, B: Array, v_cart: Array, /) -> Array:
    """Pull back Cartesian components into rep-frame components."""
    if isinstance(metric, MinkowskiMetric):
        eta = jnp.diag(jnp.array(metric.signature))
        tmp = jnp.einsum("ij,...j->...i", eta, v_cart)
        tmp2 = jnp.einsum("ji,...j->...i", B, tmp)
        return jnp.einsum("ij,...j->...i", eta, tmp2)

    return jnp.einsum("...ji,...j->...i", B, v_cart)
