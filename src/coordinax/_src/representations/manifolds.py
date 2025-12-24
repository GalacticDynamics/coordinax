"""Non-Euclidean manifold representations."""
# ruff: noqa: E501

__all__ = ("TwoSphere", "twosphere")

from collections.abc import Mapping
from typing import Any, Final, Literal as L, TypeAlias, final

import plum

import unxt as u

from . import checks
from .euclidean import Abstract2D, AbstractFixedComponentsRep, Cart2D

Ang: TypeAlias = L["angle"]

_0d = u.Angle(0, "rad")
_pid = u.Angle(180, "deg")


# -----------------------------------------------
# TwoSphere

TwoSphereKeys = tuple[L["theta"], L["phi"]]
TwoSphereDims = tuple[Ang, Ang]


@final
class TwoSphere(AbstractFixedComponentsRep[TwoSphereKeys, TwoSphereDims], Abstract2D):
    r"""Intrinsic chart on the unit two-sphere with components ``(theta, phi)``.

    Mathematical definition
    -----------------------
    .. math::
       x = \sin\theta\cos\phi,\quad y = \sin\theta\sin\phi,\quad z = \cos\theta
       \\
       \theta \in [0,\pi],\quad \phi \in (-\pi,\pi]

    Parameters
    ----------
    theta
        Polar (colatitude) angle with angular units.
    phi
        Azimuthal angle with angular units.

    Returns
    -------
    Rep
        Representation with components ``("theta","phi")`` and dimensions
        ``("angle","angle")``.

    Notes
    -----
    - ``TwoSphere`` is a curved 2D manifold; there is no global Cartesian 2D rep.
      ``cartesian_rep(TwoSphere)`` raises.
    - The intrinsic metric is ``diag(1, sin^2 theta)``.
    - The longitude is undefined at the poles ``theta=0,pi``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.r.EmbeddedManifold(
    ...     chart_kind=cx.r.twosphere,
    ...     ambient_kind=cx.r.cart3d,
    ...     params={"R": u.Quantity(2.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.r.embed_pos(rep, p)
    >>> r2 = u.uconvert("km", q["x"])**2 + u.uconvert("km", q["y"])**2 + u.uconvert("km", q["z"])**2
    >>> bool(jnp.allclose(r2.value, 4.0))
    True

    """

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["theta"], _0d, _pid)


twosphere: Final = TwoSphere()


@plum.dispatch
def cartesian_rep(obj: TwoSphere, /) -> "Cart2D":
    msg = (
        "TwoSphere has no global Cartesian 2D representation. Use an embedding via "
        "EmbeddedManifold(chart_kind=twosphere, ambient_kind=cart3d, params=...) "
        "or a specific local projection representation (e.g. stereographic) if/when provided."
    )
    raise NotImplementedError(msg)
