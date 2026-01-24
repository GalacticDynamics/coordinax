"""Non-Euclidean manifold representations."""

__all__ = ("TwoSphere", "twosphere")

from typing import Final, Literal as L, final  # noqa: N817

import plum

from . import checks
from .base import AbstractFixedComponentsChart
from .euclidean import Abstract2D, Cart2D
from coordinax._src.constants import Deg0, Deg180
from coordinax._src.custom_types import Ang, CsDict

# -----------------------------------------------
# TwoSphere

TwoSphereKeys = tuple[L["theta"], L["phi"]]
TwoSphereDims = tuple[Ang, Ang]


@final
class TwoSphere(AbstractFixedComponentsChart[TwoSphereKeys, TwoSphereDims], Abstract2D):
    r"""Intrinsic chart on the unit two-sphere with components ``(theta, phi)``.

    Mathematical definition:

    $$
       x = \sin\theta\cos\phi,\quad y = \sin\theta\sin\phi,\quad z = \cos\theta
       \\ \theta \in [0,\pi],\quad \phi \in (-\pi,\pi]
    $$

    Parameters
    ----------
    theta
        Polar (colatitude) angle with angular units.
    phi
        Azimuthal angle with angular units.

    Notes
    -----
    - ``TwoSphere`` is a curved 2D manifold; there is no global Cartesian 2D
      chart. ``cartesian_chartTwoSphere)`` raises.
    - The intrinsic metric is ``diag(1, sin^2 theta)``.
    - The longitude is undefined at the poles ``theta=0,pi``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> chart = cx.charts.EmbeddedManifold(
    ...     intrinsic_chart=cx.charts.twosphere,
    ...     ambient_chart=cx.charts.cart3d,
    ...     params={"R": u.Q(2.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.embeddings.embed_point(chart, p)
    >>> r2 = (u.uconvert("km", q["x"])**2 + u.uconvert("km", q["y"])**2
    ...       + u.uconvert("km", q["z"])**2)
    >>> bool(jnp.allclose(r2.value, 4.0))
    True

    """

    def check_data(self, data: CsDict, /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["theta"], Deg0, Deg180)


twosphere: Final = TwoSphere()


MSG_2SH_TO_C2D: Final = (
    "TwoSphere has no global Cartesian 2D representation. Use an embedding "
    "via EmbeddedManifold(intrinsic_chart=twosphere, ambient_chart=cart3d, "
    "params=...) or a specific local projection representation "
    "(e.g. stereographic) if/when provided."
)


@plum.dispatch
def cartesian_chart(obj: TwoSphere, /) -> Cart2D:
    raise NotImplementedError(MSG_2SH_TO_C2D)
