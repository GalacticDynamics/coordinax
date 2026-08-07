"""Built-in builders for `TimeDep` transforms."""

__all__ = ("RotationAboutAxis", "UniformTranslation")

from jaxtyping import Array, Shaped
from typing import Any, final

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
from .custom_types import CDict
from .rotate import Rotate
from .translate import Translate


@final
class RotationAboutAxis(eqx.Module):
    r"""Uniform rotation about a fixed axis: :math:`\theta(\tau) = \omega \tau + \phi`.

    All fields are pytree leaves: differentiate or `jax.vmap` over ``omega``,
    ``axis``, or ``phase`` directly (batching over multiple $\tau$ or multiple
    parameter sets comes from `jax.vmap` over the builder, not from
    broadcasting inside `__call__`).

    Parameters
    ----------
    omega : Quantity["angular frequency"]
        Angular frequency (e.g. rad/s or deg/s).
    axis : Array[float, (3,)]
        The rotation axis. Normalized internally; need not be unit length.
    phase : Quantity["angle"], optional
        Phase offset :math:`\phi` at :math:`\tau = 0`. Defaults to 0 rad.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> b = cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=jnp.array([0.0, 0.0, 1.0]))
    >>> op = cxfm.TimeDep(b)
    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> out = op(u.Q(1.0, "s"), q)
    >>> out["y"].round(3)
    Q(1., 'm')

    """

    omega: u.AbstractQuantity
    axis: Shaped[Array, "3"]
    phase: u.AbstractQuantity = u.Q(0.0, "rad")

    def __call__(self, tau: Any, /) -> Rotate:
        """Build the `Rotate` operator at time parameter ``tau``."""
        theta = jnp.asarray(u.ustrip("rad", self.omega * tau + self.phase))
        n = self.axis / jnp.linalg.vector_norm(self.axis)
        # Rodrigues' formula: R = I cos(th) + sin(th) [n]_x + (1-cos th) n n^T
        # The `0.0 * n[0]` terms keep the zero entries as functions of `axis`
        # so `K` (and hence `R`) stays differentiable w.r.t. every component
        # of `axis`, matching the style used for time-dependent R in the
        # `Rotate` docstring.
        zero = 0.0 * n[0]
        K = jnp.array(
            [
                [zero, -n[2], n[1]],
                [n[2], zero, -n[0]],
                [-n[1], n[0], zero],
            ]
        )
        eye = jnp.eye(3)
        ct, st = jnp.cos(theta), jnp.sin(theta)
        R = eye * ct + st * K + (1 - ct) * jnp.outer(n, n)
        return Rotate(R)


@final
class UniformTranslation(eqx.Module):
    r"""Uniform straight-line translation: :math:`\delta(\tau) = \dot\delta\,\tau`.

    All fields of ``rate`` are pytree leaves: differentiate or `jax.vmap`
    over any component directly.

    Parameters
    ----------
    rate : CDict
        Component dict of velocities (one per chart component).
    chart : AbstractChart
        The chart the offset is expressed in (same convention as `Translate`).

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm

    >>> rate = {"x": u.Q(3.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> b = cxfm.UniformTranslation(rate, chart=cxc.cart3d)
    >>> op = b(u.Q(2.0, "s"))
    >>> op.delta["x"]
    Q(6., 'km')

    """

    rate: CDict
    chart: cxc.AbstractChart = eqx.field(static=True)

    def __call__(self, tau: Any, /) -> Translate:
        """Build the `Translate` operator at time parameter ``tau``."""
        delta = {k: v * tau for k, v in self.rate.items()}
        return Translate(delta, chart=self.chart)
