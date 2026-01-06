"""Representations for embedded manifolds."""

__all__ = (
    "AbstractEmbedded",
    "EmbeddedManifold",
    "embed_pos",
    "project_pos",
    "embed_dif",
    "project_dif",
)

from dataclasses import KW_ONLY, dataclass, field, replace

from typing import Any, Generic, TypeVar, final

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from . import api, euclidean as r
from .custom_types import PDict
from .euclidean import AbstractRep, Cart3D
from .manifolds import TwoSphere

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
AmbT = TypeVar("AmbT", bound=AbstractRep)


class AbstractEmbedded:
    """Marker for reps that are intrinsic charts of an embedded manifold."""


@final
@dataclass(frozen=True, slots=True)
class EmbeddedManifold(AbstractRep[Ks, Ds], AbstractEmbedded, Generic[Ks, Ds, AmbT]):
    r"""Rep wrapper for intrinsic coordinates on an embedded manifold.

    Mathematical definition
    -----------------------
    .. math::
       \iota: U \subset M \to \mathbb{R}^{n}, \qquad q \mapsto x(q)
       \\
       \text{rep components} = (q^1, \ldots, q^k), \quad \text{ambient components} = (x^1,\ldots,x^n)

    Parameters
    ----------
    chart_kind
        Intrinsic chart rep providing component names and dimensions, e.g.
        ``("theta", "phi")`` with ``("angle", "angle")`` for ``TwoSphere``.
    ambient_kind
        Ambient position rep providing component names and dimensions, e.g.
        ``("x", "y", "z")`` with ``("length", "length", "length")`` for ``Cart3D``.
    params
        Optional embedding-specific parameters, e.g. ``{"R": Quantity(...)}`
        for a length scale in ``TwoSphere -> Cart3D``.

    Notes
    -----
    - ``components`` and ``coord_dimensions`` are inherited from ``chart_kind``.
    - Singularities are those of the intrinsic chart (e.g. poles on ``TwoSphere``).
    - Physical components are defined in orthonormal frames; coordinate
      derivatives are not represented by this rep.

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

    chart_kind: AbstractRep[Ks, Ds]
    ambient_kind: AmbT
    _: KW_ONLY
    params: PDict = field(default_factory=dict)

    @property
    def components(self) -> Ks:
        return self.chart_kind.components

    @property
    def coord_dimensions(self) -> Ds:
        return self.chart_kind.coord_dimensions

    def check_data(self, data: Any, /) -> None:
        self.chart_kind.check_data(data)

    def __hash__(self) -> int:
        params_key = tuple(sorted((k, repr(v)) for k, v in self.params.items()))
        return hash((self.__class__, self.chart_kind, self.ambient_kind, params_key))


@plum.dispatch
def cartesian_rep(obj: EmbeddedManifold, /) -> AbstractRep:
    """Return the ambient Cartesian representation for an embedded manifold."""
    ambient_cart = obj.ambient_kind.cartesian
    if isinstance(ambient_cart, Cart3D):
        return ambient_cart
    return replace(obj, ambient_kind=ambient_cart)


@plum.dispatch
def embed_pos(embedded: EmbeddedManifold, p_pos: PDict, /) -> PDict:
    r"""Embed intrinsic position coordinates into ambient coordinates.

    Mathematical definition
    -----------------------
    .. math::
       x = x(q), \qquad q = (q^1,\ldots,q^k)
       \\
       \text{For } S^2 \subset \mathbb{R}^3:\;
       x = R\sin\theta\cos\phi,\;
       y = R\sin\theta\sin\phi,\;
       z = R\cos\theta

    Parameters
    ----------
    embedded
        Embedded manifold rep with intrinsic components (e.g. ``("theta","phi")``).
    p_pos
        Intrinsic coordinates keyed by ``embedded.components``. For ``TwoSphere``,
        ``theta`` and ``phi`` must have angle units.

    Returns
    -------
    PDict
        Ambient coordinates keyed by ``embedded.ambient_kind.components``.
        For ``Cart3D`` the components are ``("x","y","z")`` with length units.

    Notes
    -----
    - ``params["R"]`` provides the length scale for ``TwoSphere -> Cart3D``.
      If absent, a ``ValueError`` is raised because ``Cart3D`` requires length units.
    - This is a position map; it does not encode physical components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.r.EmbeddedManifold(
    ...     chart_kind=cx.r.twosphere,
    ...     ambient_kind=cx.r.cart3d,
    ...     params={"R": u.Quantity(3.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.r.embed_pos(rep, p)
    >>> r2 = u.uconvert("km", q["x"])**2 + u.uconvert("km", q["y"])**2 + u.uconvert("km", q["z"])**2
    >>> bool(jnp.allclose(r2.value, 9.0))
    True

    """
    return embed_pos_chart(
        embedded.chart_kind, embedded.ambient_kind, p_pos, embedded.params
    )


@plum.dispatch
def embed_pos_chart(
    chart_kind: r.AbstractRep,
    ambient_kind: r.AbstractRep,
    p_pos: PDict,
    params: PDict,
    /,
) -> PDict:
    msg = (
        "No embed_pos rule registered for "
        f"chart={type(chart_kind)!r} into ambient={type(ambient_kind)!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def embed_pos_chart(
    chart_kind: TwoSphere,
    ambient_kind: Cart3D,
    p_pos: PDict,
    params: PDict,
    /,
) -> PDict:
    del chart_kind, ambient_kind
    R = params.get("R")
    R = eqx.error_if(
        R,
        R is None,
        "TwoSphere -> Cart3D embedding requires params['R'] for length scale.",
    )
    if u.dimension_of(R) != u.dimension("length"):
        msg = "params['R'] must have length units for TwoSphere -> Cart3D embedding."
        raise ValueError(msg)

    theta = u.ustrip(AllowValue, "rad", p_pos["theta"])
    phi = u.ustrip(AllowValue, "rad", p_pos["phi"])
    x = R * jnp.sin(theta) * jnp.cos(phi)
    y = R * jnp.sin(theta) * jnp.sin(phi)
    z = R * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def project_pos(embedded: EmbeddedManifold, p_ambient: PDict, /) -> PDict:
    r"""Project ambient coordinates onto intrinsic chart coordinates.

    Mathematical definition
    -----------------------
    .. math::
       q = \pi(x), \qquad x \in \mathbb{R}^n
       \\
       \text{For } S^2:\;
       \theta = \arccos\!\left(\frac{z}{r}\right),\;
       \phi = \operatorname{atan2}(y,x),\;
       r = \sqrt{x^2+y^2+z^2}

    Parameters
    ----------
    embedded
        Embedded manifold rep with ambient components (e.g. ``("x","y","z")``).
    p_ambient
        Ambient coordinates keyed by ``embedded.ambient_kind.components`` with
        length units for ``Cart3D``.

    Returns
    -------
    PDict
        Intrinsic coordinates keyed by ``embedded.components``; for ``TwoSphere``,
        ``theta`` and ``phi`` are angles.

    Notes
    -----
    - Inputs are normalized by ``r`` so near-sphere points are accepted.
    - At the poles where ``sin(theta)=0``, ``phi`` is set to 0 by convention.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.r.EmbeddedManifold(
    ...     chart_kind=cx.r.twosphere,
    ...     ambient_kind=cx.r.cart3d,
    ...     params={"R": u.Quantity(1.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.r.embed_pos(rep, p)
    >>> p2 = cx.r.project_pos(rep, q)
    >>> bool(jnp.allclose(u.uconvert("rad", p2["theta"]).value, u.uconvert("rad", p["theta"]).value))
    True

    """
    return project_pos_chart(
        embedded.chart_kind, embedded.ambient_kind, p_ambient, embedded.params
    )


@plum.dispatch
def project_pos_chart(
    chart_kind: r.AbstractRep,
    ambient_kind: r.AbstractRep,
    p_ambient: PDict,
    params: PDict,
    /,
) -> PDict:
    msg = (
        "No project_pos rule registered for "
        f"chart={type(chart_kind)!r} from ambient={type(ambient_kind)!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def project_pos_chart(
    chart_kind: TwoSphere,
    ambient_kind: r.Cart3D,
    p_ambient: PDict,
    params: PDict,
    /,
) -> PDict:
    del chart_kind, ambient_kind, params
    x = u.ustrip(AllowValue, p_ambient["x"])
    y = u.ustrip(AllowValue, p_ambient["y"])
    z = u.ustrip(AllowValue, p_ambient["z"])

    r_val = jnp.sqrt(x**2 + y**2 + z**2)
    eps = jnp.asarray(1e-12, dtype=r_val.dtype)
    safe_r = jnp.where(r_val > eps, r_val, 1.0)

    cos_theta = jnp.clip(z / safe_r, -1.0, 1.0)
    theta = u.Quantity(jnp.arccos(cos_theta), "rad")

    rho_val = jnp.hypot(x, y)
    safe_rho = rho_val > eps
    phi_val = jnp.atan2(y, x)
    phi = u.Quantity(jnp.where(safe_rho, phi_val, 0.0), "rad")

    return {"theta": theta, "phi": phi}


@plum.dispatch
def embed_dif(embedded: EmbeddedManifold, p_dif: PDict, p_pos: PDict, /) -> PDict:
    r"""Embed intrinsic physical components into ambient physical components.

    Mathematical definition
    -----------------------
    .. math::
       B(q) = [\hat e_1(q)\ \cdots\ \hat e_k(q)], \qquad v_{\text{cart}} = B(q)\,v_{\text{rep}}
       \\
       \hat e_i \text{ are orthonormal w.r.t. the ambient metric}

    Parameters
    ----------
    embedded
        Embedded manifold rep defining the intrinsic components.
    p_dif
        Physical components keyed by ``embedded.components`` with uniform units
        (all speed or all acceleration).
    p_pos
        Intrinsic position coordinates used to evaluate the tangent frame.

    Returns
    -------
    PDict
        Ambient physical components keyed by ``embedded.ambient_kind.components``.

    Notes
    -----
    - These are physical components, not coordinate derivatives.
    - The ambient metric is used for orthonormality.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.r.EmbeddedManifold(
    ...     chart_kind=cx.r.twosphere,
    ...     ambient_kind=cx.r.cart3d,
    ...     params={"R": u.Quantity(1.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Quantity(1.0, "km/s"), "phi": u.Quantity(0.0, "km/s")}
    >>> v_cart = cx.r.embed_dif(rep, v_tan, p)
    >>> bool(jnp.allclose(u.uconvert("km/s", v_cart["z"]).value, -1.0))
    True

    """
    return api.diff_map(embedded.ambient_kind, embedded, p_dif, p_pos)


@plum.dispatch
def project_dif(
    embedded: EmbeddedManifold, p_dif_ambient: PDict, p_pos: PDict, /
) -> PDict:
    r"""Project ambient physical components onto the manifold tangent space.

    Mathematical definition
    -----------------------
    .. math::
       v_{\text{rep}} = B(q)^{\mathsf{T}} v_{\text{cart}}
       \\
       \text{(Euclidean ambient metric; normal component is discarded)}

    Parameters
    ----------
    embedded
        Embedded manifold rep defining the intrinsic components.
    p_dif_ambient
        Ambient physical components keyed by ``embedded.ambient_kind.components``.
    p_pos
        Intrinsic position coordinates used to evaluate the tangent frame.

    Returns
    -------
    PDict
        Tangent-space physical components keyed by ``embedded.components``.

    Notes
    -----
    - This is an orthogonal projection onto the tangent space.
    - Inputs must be physical components, not coordinate derivatives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.r.EmbeddedManifold(
    ...     chart_kind=cx.r.twosphere,
    ...     ambient_kind=cx.r.cart3d,
    ...     params={"R": u.Quantity(1.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Quantity(1.0, "km/s"), "phi": u.Quantity(0.0, "km/s")}
    >>> v_cart = cx.r.embed_dif(rep, v_tan, p)
    >>> v_back = cx.r.project_dif(rep, v_cart, p)
    >>> bool(jnp.allclose(u.uconvert("km/s", v_back["theta"]).value, 1.0))
    True

    """
    p_pos_ambient = embed_pos(embedded, p_pos)
    return api.diff_map(embedded, embedded.ambient_kind, p_dif_ambient, p_pos_ambient)
