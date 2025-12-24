"""Representations for embedded manifolds."""

__all__ = (
    "AbstractEmbedded",
    "EmbeddedManifold",
    # "embed_point",  # public from api.py
    # "project_point",  # public from api.py
    # "embed_tangent",  # public from api.py
    # "project_tangent",  # public from api.py
)

from dataclasses import KW_ONLY, dataclass, field, replace

from collections.abc import Mapping
from typing import Any, Generic, TypeVar, final

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax._src.charts as cxc
from coordinax._src import api
from coordinax._src.custom_types import CsDict, Ds, Ks
from coordinax._src.utils import uconvert_to_rad

AmbT = TypeVar("AmbT", bound=cxc.AbstractChart[Any, Any])


class AbstractEmbedded:
    """Marker for reps that are intrinsic charts of an embedded manifold."""


@final
@dataclass(frozen=True, slots=True)
class EmbeddedManifold(
    cxc.AbstractChart[Ks, Ds], AbstractEmbedded, Generic[Ks, Ds, AmbT]
):
    r"""Rep wrapper for intrinsic coordinates on an embedded manifold.

    Mathematical definition:
    $$
       \iota: U \subset M \to \mathbb{R}^{n}, \qquad q \mapsto x(q) \\ \text{rep
       components} = (q^1, \ldots, q^k), \quad \text{ambient components} =
       (x^1,\ldots,x^n)
    $$

    Parameters
    ----------
    intrinsic_chart
        Intrinsic chart chart providing component names and dimensions, e.g.
        ``("theta", "phi")`` with ``("angle", "angle")`` for
        {class}`coordinax.charts.TwoSphere`.
    ambient_chart
        Ambient position chart providing component names and dimensions, e.g.
        ``("x", "y", "z")`` with ``("length", "length", "length")`` for
        {class}`coordinax.charts.Cart3D`.
    params
        Optional embedding-specific parameters, e.g. ``{"R": Quantity(...)}` for
        a length scale in {class}`coordinax.charts.TwoSphere` ->
        {class}`coordinax.charts.Cart3D`.

    Notes
    -----
    - ``components`` and ``coord_dimensions`` are inherited from ``intrinsic_chart``.
    - Singularities are those of the intrinsic chart (e.g. poles on
      {class}`coordinax.charts.TwoSphere`).
    - Physical components are defined in orthonormal frames; coordinate
      derivatives are not represented by this rep.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.charts.EmbeddedManifold(
    ...     intrinsic_chart=cx.charts.twosphere,
    ...     ambient_chart=cx.charts.cart3d,
    ...     params={"R": u.Q(2.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.embeddings.embed_point(rep, p)
    >>> r2 = (q["x"].uconvert("km")**2 + q["y"].uconvert("km")**2
    ...       + q["z"].uconvert("km")**2)
    >>> bool(jnp.allclose(r2.value, 4.0))
    True

    """

    intrinsic_chart: cxc.AbstractChart[Ks, Ds]
    ambient_chart: AmbT
    _: KW_ONLY
    params: CsDict = field(default_factory=dict)

    @property
    def components(self) -> Ks:
        return self.intrinsic_chart.components

    @property
    def coord_dimensions(self) -> Ds:
        return self.intrinsic_chart.coord_dimensions

    def check_data(self, data: Any, /) -> None:
        self.intrinsic_chart.check_data(data)

    def __hash__(self) -> int:
        params_key = tuple(sorted((k, repr(v)) for k, v in self.params.items()))
        return hash(
            (self.__class__, self.intrinsic_chart, self.ambient_chart, params_key)
        )


@plum.dispatch
def cartesian_chart(obj: EmbeddedManifold, /) -> cxc.AbstractChart:  # type: ignore[type-arg]
    """Return the ambient Cartesian representation for an embedded manifold."""
    ambient_cart = obj.ambient_chart.cartesian
    if isinstance(ambient_cart, cxc.Cart3D):
        return ambient_cart
    return replace(obj, ambient_chart=ambient_cart)


# ===================================================================
# Point embedding


@plum.dispatch
def embed_point(
    embedded: EmbeddedManifold,  # type: ignore[type-arg]
    p_pos: CsDict,
    /,
    *,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    r"""Embed intrinsic point coordinates into ambient coordinates.

    Mathematical definition:
    $$
       x = x(q), \qquad q = (q^1,\ldots,q^k)
       \\
       \text{For } S^2 \subset \mathbb{R}^3:\;
       x = R\sin\theta\cos\phi,\;
       y = R\sin\theta\sin\phi,\;
       z = R\cos\theta
    $$

    Parameters
    ----------
    embedded
        Embedded manifold rep with intrinsic components (e.g. ``("theta","phi")``).
    p_pos
        Intrinsic coordinates keyed by ``embedded.components``. For ``TwoSphere``,
        ``theta`` and ``phi`` must have angle units.
    usys
        Unit system for input quantities. Only needed if input is raw arrays,
        not {class}`~unxt.AbstractQuantity` objects.

    Returns
    -------
    CsDict
        Ambient coordinates keyed by ``embedded.ambient_chart.components``.
        For ``Cart3D`` the components are ``("x","y","z")`` with length units.

    Notes
    -----
    - ``params["R"]`` provides the length scale for ``TwoSphere -> Cart3D``.
      If absent, a ``ValueError`` is raised because ``Cart3D`` requires length units.
    - This is a point map; it does not encode physical components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.charts.EmbeddedManifold(
    ...     intrinsic_chart=cx.charts.twosphere,
    ...     ambient_chart=cx.charts.cart3d,
    ...     params={"R": u.Q(3.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.embeddings.embed_point(rep, p)
    >>> r2 = (
    ...     u.uconvert("km", q["x"])**2 +
    ...     u.uconvert("km", q["y"])**2 +
    ...     u.uconvert("km", q["z"])**2
    ... )
    >>> bool(jnp.allclose(r2.value, 9.0))
    True

    """
    return api.embed_point(
        embedded.intrinsic_chart,
        embedded.ambient_chart,
        p_pos,
        embedded.params,
        usys=usys,
    )


@plum.dispatch
def embed_point(
    intrinsic_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    ambient_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p_pos: CsDict,
    params: Mapping[str, Any],
    /,
    *,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    msg = (
        "No embed_point rule registered for "
        f"chart={type(intrinsic_chart)!r} into ambient={type(ambient_chart)!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def embed_point(
    intrinsic_chart: cxc.TwoSphere,
    ambient_chart: cxc.Cart3D,
    p_pos: CsDict,
    params: Mapping[str, Any],
    /,
    *,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    del intrinsic_chart, ambient_chart
    R = params.get("R")
    R = eqx.error_if(
        R,
        R is None,
        "TwoSphere -> Cart3D embedding requires params['R'] for length scale.",
    )
    if u.dimension_of(R) != u.dimension("length"):
        msg = "params['R'] must have length units for TwoSphere -> Cart3D embedding."
        raise ValueError(msg)

    theta = uconvert_to_rad(p_pos["theta"], usys)
    phi = uconvert_to_rad(p_pos["phi"], usys)
    stheta = jnp.sin(theta)
    x = R * stheta * jnp.cos(phi)
    y = R * stheta * jnp.sin(phi)
    z = R * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


# ===================================================================
# Point projection


@plum.dispatch
def project_point(
    embedded: EmbeddedManifold,  # type: ignore[type-arg]
    p_ambient: CsDict,
    /,
    *,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    r"""Project ambient coordinates onto intrinsic chart coordinates.

    Mathematical definition:
    $$
       q = \pi(x), \qquad x \in \mathbb{R}^n
       \\
       \text{For } S^2:\;
       \theta = \arccos\!\left(\frac{z}{r}\right),\;
       \phi = \operatorname{atan2}(y,x),\;
       r = \sqrt{x^2+y^2+z^2}
    $$

    Parameters
    ----------
    embedded
        Embedded manifold rep with ambient components (e.g. ``("x","y","z")``).
    p_ambient
        Ambient coordinates keyed by ``embedded.ambient_chart.components`` with
        length units for ``Cart3D``.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
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
    >>> rep = cx.charts.EmbeddedManifold(
    ...     intrinsic_chart=cx.charts.twosphere,
    ...     ambient_chart=cx.charts.cart3d,
    ...     params={"R": u.Q(1.0, "km")},
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cx.embeddings.embed_point(rep, p)
    >>> p2 = cx.embeddings.project_point(rep, q)
    >>> bool(
    ...     jnp.allclose(
    ...         u.uconvert("rad", p2["theta"]).value,
    ...         u.uconvert("rad", p["theta"]).value,
    ...     )
    ... )
    True

    """
    return api.project_point(
        embedded.intrinsic_chart,
        embedded.ambient_chart,
        p_ambient,
        embedded.params,
        usys=usys,
    )


@plum.dispatch
def project_point(
    intrinsic_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    ambient_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p_ambient: CsDict,
    params: Mapping[str, Any],
    /,
    *,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    msg = (
        "No project_point rule registered for "
        f"chart={type(intrinsic_chart)!r} from ambient={type(ambient_chart)!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def project_point(
    intrinsic_chart: cxc.TwoSphere,
    ambient_chart: cxc.Cart3D,
    p_ambient: CsDict,
    params: Mapping[str, Any],
    /,
    *,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    del intrinsic_chart, ambient_chart, params, usys
    x, y, z = p_ambient["x"], p_ambient["y"], p_ambient["z"]

    r = jnp.sqrt(x**2 + y**2 + z**2)
    r_val = u.ustrip(AllowValue, r)
    eps = jnp.asarray(1e-12, dtype=r_val.dtype)
    safe_r = jnp.where(r_val > eps, r, jnp.ones_like(1.0))

    cos_theta = jnp.clip(z / safe_r, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    rho_val = jnp.hypot(x, y)
    safe_rho = rho_val > jnp.full_like(rho_val, eps)
    phi_val = jnp.atan2(y, x)
    phi = jnp.where(safe_rho, phi_val, 0.0)

    return {"theta": theta, "phi": phi}


# ===================================================================
# Tangent embedding


@plum.dispatch
def embed_tangent(
    embedded: EmbeddedManifold,  # type: ignore[type-arg]
    v_chart: CsDict,
    /,
    *,
    at: CsDict,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    r"""Embed intrinsic physical tangent components into ambient physical components.

    Mathematical definition:
    $$
       B(q) = [\hat e_1(q)\ \cdots\ \hat e_k(q)], \qquad
       v_{\text{cart}} = B(q)\,v_{\text{rep}}
       \\
       \hat e_i \text{ are orthonormal w.r.t. the ambient metric}
    $$

    Parameters
    ----------
    embedded
        Embedded manifold rep defining the intrinsic components.
    v_chart
        Physical tangent components (uniform units, not coordinate derivatives)
        keyed by ``embedded.components`` (all speed or all acceleration).
    at
        Intrinsic point coordinates used to evaluate the tangent frame.
    usys
        Unit system for input quantities. Only needed if input is raw arrays,
        not {class}`~unxt.AbstractQuantity` objects.

    Returns
    -------
    CsDict
        Ambient physical tangent components keyed by
        ``embedded.ambient_chart.components``.

    Notes
    -----
    - These are physical tangent components with uniform units,
      not coordinate derivatives.
    - The ambient metric is used for orthonormality.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> rep = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                            params={"R": u.Q(1.0, "km")})
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}
    >>> v_cart = cxe.embed_tangent(rep, v_tan, at=p)
    >>> bool(jnp.allclose(u.uconvert("km/s", v_cart["z"]).value, -1.0))
    True

    """
    return api.physical_tangent_transform(
        embedded.ambient_chart, embedded, v_chart, at=at, usys=usys
    )


# ===================================================================
# Tangent Projection


@plum.dispatch
def project_tangent(
    embedded: EmbeddedManifold,  # type: ignore[type-arg]
    v_ambient: CsDict,
    /,
    *,
    at: CsDict,
    usys: u.AbstractUnitSystem | None = None,
) -> CsDict:
    r"""Project ambient physical components onto the manifold tangent space.

    Mathematical definition:
    $$
       v_{\text{rep}} = B(q)^{\mathsf{T}} v_{\text{cart}}
       \\
       \text{(Euclidean ambient metric; normal component is discarded)}
    $$

    Parameters
    ----------
    embedded
        Embedded manifold rep defining the intrinsic components.
    v_ambient
        Ambient physical tangent components (not coordinate derivatives)
        keyed by ``embedded.ambient_chart.components``.
    at
        Intrinsic point coordinates used to evaluate the tangent frame.
    usys
        Unit system for input quantities. Only needed if input is raw arrays,
        not {class}`~unxt.AbstractQuantity` objects.

    Returns
    -------
    CsDict
        Tangent-space physical components keyed by ``embedded.components``.

    Notes
    -----
    - This is an orthogonal projection (ambient Euclidean metric) onto
      the tangent space.
    - Inputs must be physical tangent components with uniform units,
      not coordinate derivatives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> rep = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                            params={"R": u.Q(1.0, "km")})
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}
    >>> v_cart = cxe.embed_tangent(rep, v_tan, at=p)
    >>> v_back = cxe.project_tangent(rep, v_cart, at=p)
    >>> bool(jnp.allclose(u.uconvert("km/s", v_back["theta"]).value, 1.0))
    True

    """
    p_pos_ambient = api.embed_point(embedded, at, usys=usys)
    return api.physical_tangent_transform(
        embedded, embedded.ambient_chart, v_ambient, at=p_pos_ambient, usys=usys
    )
