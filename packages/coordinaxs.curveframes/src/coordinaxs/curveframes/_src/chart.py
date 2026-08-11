r"""A chart on a tubular neighbourhood of a curve.

Coordinates are $(\tau, n_1, n_2)$:

$$ \mathbf{x} = \boldsymbol{\gamma}(\tau)
   + n_1\mathbf{U}_1(\tau) + n_2\mathbf{U}_2(\tau) $$

where $(\mathbf{T},\mathbf{U}_1,\mathbf{U}_2)$ is the triad supplied by an
`AbstractCurveFrameBuilder`. The same class serves Frenet--Serret and Bishop;
they differ only in the builder handed in.

Note that $\tau$ is the *curve parameter*, not arc length. The builders are
$\tau$-parameterised, so $g_{\tau\tau}$ carries a $\|\gamma'\|^2$ speed factor
rather than reducing to $(1-k_1n_1-k_2n_2)^2$.
"""

__all__ = ("TubularChart",)

import dataclasses

from typing import Any, ClassVar, final, override

import equinox as eqx
import jax
import jax.numpy as jnp

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import unxt as u
from coordinax._src.base import AbstractParameterizedChart

from .base import AbstractCurveFrameBuilder


@final
class TubularChart(AbstractParameterizedChart):
    r"""Chart on a tubular neighbourhood of a curve.

    Differentiability is opt-in per instance, exactly as for any parameterized
    chart: a curve that is an `equinox.Module` holding `unxt.Quantity`
    parameters contributes leaves and can be differentiated through; a plain
    function closes over trace-time constants and contributes none.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> chart = cxfc.TubularChart(
    ...     cxfc.BishopBuilder(circle),
    ...     tau_bounds=(u.Q(-1.0, "s"), u.Q(7.0, "s")),
    ... )
    >>> chart.components
    ('tau', 'n1', 'n2')
    >>> chart.coord_dimensions
    ('time', 'length', 'length')

    """

    builder: AbstractCurveFrameBuilder
    """The curve-frame builder supplying gamma and the triad."""

    _: dataclasses.KW_ONLY

    tau_bounds: tuple[Any, Any]
    """Scan range for the inverse solve.

    Must cover the curve of interest, and for a **closed** curve must not span
    more than one period: `gamma(tau)` and `gamma(tau + period)` are the same
    ambient point, so a wider range makes the nearest-point solve an exact tie
    and the recovered `tau` arbitrary between the two.

    The endpoints of a one-period range still coincide for a closed curve;
    the scan's tie-break resolves that seam to the lower bound.
    """

    n_seed: int = eqx.field(static=True, default=64)
    """Seed points for the inverse scan. Static, since it is a loop bound."""

    M: ClassVar[Any]

    @override
    @property
    def M(self) -> Any:
        """The ambient manifold, always flat 3-space regardless of the curve."""
        return cxm.R3

    @property
    def components(self) -> tuple[str, str, str]:
        return ("tau", "n1", "n2")

    @property
    def coord_dimensions(self) -> tuple[str, str, str]:
        # The first coordinate inherits whatever the curve is parameterised by,
        # so this cannot be a class-level tuple the way most charts declare it.
        return (str(u.dimension_of(self.builder.tau_unit)), "length", "length")

    @property
    def cartesian(self) -> cxc.Cart3D:
        return cxc.cart3d

    def check_data(self, data: dict, /, *, values: bool = False, **kw: Any) -> dict:
        # Forward `values`: the base class gates its coordinate-dimension check
        # on it, and binding it as a named parameter keeps it out of `**kw`.
        super().check_data(data, values=values, **kw)
        if values:
            # Inside the reach the Jacobian factor is positive; at the focal
            # distance it vanishes and the coordinates stop being unique.
            #
            # Hybrid form, copied from `_src/charts/checks.py`: `eqx.error_if`
            # under trace (a Python `bool` on a tracer raises
            # `TracerBoolConversionError`), a plain `ValueError` when concrete.
            # Do not collapse it to one branch.
            pred = jnp.any(self.jacobian_factor(data) <= 0)
            msg = (
                "point lies outside the reach of the curve: the tubular "
                "coordinates are not injective there"
            )
            if isinstance(pred, jax.core.Tracer):
                # The return value MUST be threaded back into the data that is
                # returned. `eqx.error_if` is eliminated as dead code when its
                # result goes unused -- a bare `eqx.error_if(pred, pred, msg)`
                # compiles away and the guard silently passes under `jit`
                # (verified: it returned n1=-1.6, well outside the reach).
                data = {**data, "n1": eqx.error_if(data["n1"], pred, msg)}
            elif bool(pred):
                raise ValueError(msg)
        return data

    def jacobian_factor(self, data: dict, /) -> Any:
        r"""$\partial\mathbf{x}/\partial\tau$ scaled by the on-curve speed.

        Equals $1-k_1n_1-k_2n_2$ for a unit-speed curve; in general it is that
        quantity times $\|\gamma'\|$. It is positive inside the reach and
        vanishes at the focal distance, which is the test that matters.
        """
        tau, n1, n2 = data["tau"], data["n1"], data["n2"]
        unit = self.builder.tau_unit

        def gamma_v(t: jax.Array) -> jax.Array:
            return jnp.asarray(
                self.builder.location(u.Q(t, unit)).ustrip("km"), dtype=float
            )

        def offset_v(t: jax.Array) -> jax.Array:
            R = self.builder.rotation_matrix(u.Q(t, unit))
            n1_v = jnp.asarray(n1.ustrip("km"), dtype=float)
            n2_v = jnp.asarray(n2.ustrip("km"), dtype=float)
            return gamma_v(t) + n1_v * R[1] + n2_v * R[2]

        tau_v = jnp.asarray(tau.ustrip(unit), dtype=float)
        dx = jax.jacfwd(offset_v)(tau_v)
        speed = jnp.linalg.norm(jax.jacfwd(gamma_v)(tau_v))
        # Project onto the tangent: dx is parallel to T, and past the focal
        # distance it REVERSES. `norm(dx)` is sign-blind and bounces back up
        # (measured: at n1=-1.1 on the unit circle, norm=+0.1 but the true
        # factor is -0.1), so a `<= 0` guard built on the norm can only fire
        # exactly at the focal point -- a measure-zero set it will never hit.
        T = self.builder.rotation_matrix(u.Q(tau_v, unit))[0]
        return jnp.dot(dx, T) / speed
