"""Orthonormal frame utilities for representations."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax.api as cxapi
import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax.metrics as cxm
from coordinax._src.custom_types import OptUSys
from coordinax._src.utils import uconvert_to_rad
from coordinax.api import CsDict

#####################################################################
# Orthonormal Frame in Cartesian Components


@plum.dispatch
def frame_cart(
    chart: cxc.Cart1D | cxc.Cart2D | cxc.Cart3D | cxc.CartND,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    del at, usys
    return jnp.eye(chart.ndim)


@plum.dispatch
def frame_cart(
    chart: cxc.Cylindrical3D, /, *, at: CsDict, usys: OptUSys = None
) -> Array:
    """Cylindrical orthonormal frame (e_rho, e_phi, e_z) in Cartesian components."""
    del chart
    phi = u.ustrip(AllowValue, uconvert_to_rad(at["phi"], usys))
    s, c = jnp.sin(phi), jnp.cos(phi)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@plum.dispatch
def frame_cart(chart: cxc.Spherical3D, /, *, at: CsDict, usys: OptUSys = None) -> Array:
    """Spherical (physics convention) orthonormal frame in Cartesian components."""
    del chart
    theta = u.ustrip(AllowValue, uconvert_to_rad(at["theta"], usys))
    phi = u.ustrip(AllowValue, uconvert_to_rad(at["phi"], usys))

    st, ct = jnp.sin(theta), jnp.cos(theta)
    sp, cp = jnp.sin(phi), jnp.cos(phi)

    e_r = jnp.array([st * cp, st * sp, ct])
    e_th = jnp.array([ct * cp, ct * sp, -st])
    e_ph = jnp.array([-sp, cp, 0])
    return jnp.stack([e_r, e_th, e_ph], axis=-1)


@plum.dispatch
def frame_cart(
    chart: cxc.LonLatSpherical3D, /, *, at: CsDict, usys: OptUSys = None
) -> Array:
    """Lon/Lat spherical orthonormal frame in Cartesian components."""
    del chart
    lat = u.ustrip(AllowValue, uconvert_to_rad(at["lat"], usys))
    lon = u.ustrip(AllowValue, uconvert_to_rad(at["lon"], usys))
    theta = (jnp.pi / 2) - lat
    sph_pos = {"r": at["distance"], "theta": theta, "phi": lon}
    return cxapi.frame_cart(cxc.sph3d, at=sph_pos, usys=usys)


@plum.dispatch
def frame_cart(
    chart: cxc.MathSpherical3D, /, *, at: CsDict, usys: OptUSys = None
) -> Array:
    """MathSpherical3D orthonormal frame in Cartesian components."""
    del chart
    sph_pos = {"r": at["r"], "theta": at["phi"], "phi": at["theta"]}
    return cxapi.frame_cart(cxc.sph3d, at=sph_pos, usys=usys)


@plum.dispatch
def frame_cart(
    chart: cxc.LonCosLatSpherical3D, /, *, at: CsDict, usys: OptUSys = None
) -> Array:
    """Lon*cos(lat), lat, distance spherical orthonormal frame in Cartesian."""
    del chart
    lon_coslat = at["lon_coslat"]
    lat = u.ustrip(AllowValue, uconvert_to_rad(at["lat"], usys))
    dist = at["distance"]

    cos_lat = jnp.cos(lat)

    eps = jnp.asarray(1e-12, dtype=cos_lat.dtype)
    safe = jnp.abs(cos_lat) > eps

    lon = lon_coslat / jnp.where(safe, cos_lat, 1.0)
    lon0 = u.Q(0, u.unit_of(lon_coslat))
    lon = jnp.where(safe, lon, lon0)

    theta = (jnp.pi / 2) - lat
    sph_pos = {"r": dist, "theta": theta, "phi": lon}
    return cxapi.frame_cart(cxc.sph3d, at=sph_pos, usys=usys)


# ---------------------------------------------------------
# Embedded Manifolds


@plum.dispatch
def frame_cart(
    chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    """Return the orthonormal tangent frame in ambient Cartesian components.

    Redispatches to (`intrinsic_chart`, `ambient_chart`, at=at, usys=usys).

    """
    return cxapi.frame_cart(
        chart.intrinsic_chart, chart.ambient_chart, at=at, usys=usys
    )


@plum.dispatch
def frame_cart(
    intrinsic_chart: cxc.TwoSphere,
    ambient_chart: cxc.Cart3D,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    """Return the orthonormal tangent frame in ambient Cartesian components."""
    theta = u.ustrip(AllowValue, uconvert_to_rad(at["theta"], usys))
    phi = u.ustrip(AllowValue, uconvert_to_rad(at["phi"], usys))

    ctheta = jnp.cos(theta)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    e_theta = jnp.array([ctheta * cphi, ctheta * sphi, -jnp.sin(theta)])
    e_phi = jnp.array([-sphi, cphi, jnp.zeros_like(phi)])

    return jnp.stack([e_theta, e_phi], axis=-1)


# ---------------------------------------------------------
# Product Charts


@plum.dispatch
def frame_cart(
    chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    """Product chart orthonormal frame (block diagonal of factor frames).

    Constructs the frame matrix as a block diagonal of factor frames.

    Mathematical definition:
        B(p) = diag(B₁(p₁), ..., Bₖ(pₖ))

    where Bᵢ are the factor frames.
    """
    # Split the point by factors
    p_factors = chart.split_components(at)

    # Get frame matrix for each factor
    factor_frames = [
        cxapi.frame_cart(f, at=p_f, usys=usys)
        for f, p_f in zip(chart.factors, p_factors, strict=True)
    ]

    # Build block diagonal matrix manually Each factor_frames[i] has shape (...,
    # Nᵢ, nᵢ) where Nᵢ is ambient dim, nᵢ is intrinsic dim
    if len(factor_frames) == 1:
        return factor_frames[0]

    # For simplicity, use scipy.linalg.block_diag-like construction with JAX
    # Build block-diagonal by placing each block on the diagonal
    total_ambient = sum(f.shape[-2] for f in factor_frames)
    total_intrinsic = sum(f.shape[-1] for f in factor_frames)

    # Get batch shape from first frame (all should have same batch shape)
    batch_shape = factor_frames[0].shape[:-2]
    dtype = factor_frames[0].dtype

    # Initialize zero matrix
    result = jnp.zeros((*batch_shape, total_ambient, total_intrinsic), dtype=dtype)

    # Place each block on the diagonal
    row_offset = 0
    col_offset = 0
    for frame in factor_frames:
        n_ambient = frame.shape[-2]
        n_intrinsic = frame.shape[-1]
        # Use advanced indexing to place the block
        result = result.at[
            ...,
            row_offset : row_offset + n_ambient,
            col_offset : col_offset + n_intrinsic,
        ].set(frame)
        row_offset += n_ambient
        col_offset += n_intrinsic

    return result


@plum.dispatch
def frame_cart(
    chart: cxc.SpaceTimeCT,  # type: ignore[type-arg]
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    """SpaceTimeCT orthonormal frame in Cartesian (ct,x,y,z) components."""
    sp = chart.spatial_chart
    p_sp = {k: at[k] for k in sp.components}
    B_sp = cxapi.frame_cart(sp, at=p_sp, usys=usys)

    n_sp = B_sp.shape[-1]
    one = jnp.ones((1, 1), dtype=B_sp.dtype)
    z1 = jnp.zeros((1, n_sp), dtype=B_sp.dtype)
    z3 = jnp.zeros((n_sp, 1), dtype=B_sp.dtype)
    return jnp.block([[one, z1], [z3, B_sp]])


@plum.dispatch
def frame_cart(
    chart: cxc.SpaceTimeEuclidean,  # type: ignore[type-arg]
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    """SpaceTimeEuclidean orthonormal frame in Cartesian (ct,x,y,z) components."""
    sp = chart.spatial_chart
    p_sp = {k: at[k] for k in sp.components}
    B_sp = cxapi.frame_cart(sp, at=p_sp, usys=usys)

    n_sp = B_sp.shape[-1]
    one = jnp.ones((1, 1), dtype=B_sp.dtype)
    z1 = jnp.zeros((1, n_sp), dtype=B_sp.dtype)
    z3 = jnp.zeros((n_sp, 1), dtype=B_sp.dtype)
    return jnp.block([[one, z1], [z3, B_sp]])


#####################################################################
# Pushforward and Pullback


@plum.dispatch
def pushforward(frame_basis: Array, v_chart: Array, /) -> Array:
    """Push forward components from a rep frame into Cartesian components."""
    return jnp.einsum("...ij,...j->...i", frame_basis, v_chart)


@plum.dispatch
def pullback(metric: cxm.AbstractMetric, frame_basis: Array, v_cart: Array, /) -> Array:
    """Pull back Cartesian components into rep-frame components."""
    return jnp.einsum("...ji,...j->...i", frame_basis, v_cart)


@plum.dispatch
def pullback(
    metric: cxm.MinkowskiMetric, frame_basis: Array, v_cart: Array, /
) -> Array:
    """Pull back Cartesian components into rep-frame components."""
    eta = jnp.diag(jnp.array(metric.signature))
    tmp = jnp.einsum("ij,...j->...i", eta, v_cart)
    tmp2 = jnp.einsum("ji,...j->...i", frame_basis, tmp)
    return jnp.einsum("ij,...j->...i", eta, tmp2)
