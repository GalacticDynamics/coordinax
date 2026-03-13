"""Representations for embedded manifolds."""

__all__ = ()


from collections.abc import Mapping
from typing import Any

import plum

import coordinax.api.embeddings as cxeapi
import coordinax.api.tangents as cxtapi
import coordinax.charts as cxc
from .conv_chart import EmbeddedChart
from .core import EmbeddedManifold
from .sphere2in3d import TwoSphereIn3D
from coordinax.internal.custom_types import CDict, OptUSys

# ===================================================================
# Point embedding


@plum.dispatch
def embed_point(
    embedded: EmbeddedChart,  # type: ignore[type-arg]
    p_pos: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
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
        Intrinsic coordinates keyed by ``embedded.components``. For
        ``SphericalTwoSphere``, ``theta`` and ``phi`` must have angle units.
    usys
        Unit system for input quantities. Only needed if input is raw arrays,
        not {class}`~unxt.AbstractQuantity` objects.

    Returns
    -------
    CDict
        Ambient coordinates keyed by ``embedded.ambient.components``.
        For ``Cart3D`` the components are ``("x","y","z")`` with length units.

    Notes
    -----
    - The embedding object (e.g. ``TwoSphereIn3D``) owns the length scale
      (``radius``) for ``SphericalTwoSphere -> Cart3D``.
    - This is a point map; it does not encode physical components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(3.0, "km"), ambient=cxc.cart3d)
    >>> chart = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cxe.embed_point(chart, p)
    >>> r2 = (
    ...     u.uconvert("km", q["x"])**2 +
    ...     u.uconvert("km", q["y"])**2 +
    ...     u.uconvert("km", q["z"])**2
    ... )
    >>> bool(jnp.allclose(r2.value, 9.0))
    True

    """
    return cxeapi.embed_point(
        embedded.intrinsic,
        embedded.embedding,
        p_pos,
        usys=usys,
    )


@plum.dispatch
def embed_point(
    embedded: EmbeddedManifold,
    p_pos: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Embed intrinsic point coordinates into ambient coordinates (manifold)."""
    return cxeapi.embed_point(
        embedded.embedding.intrinsic, embedded.embedding, p_pos, usys=usys
    )


@plum.dispatch
def embed_point(
    intrinsic_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    ambient_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p_pos: CDict,
    params: Mapping[str, Any],
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    del p_pos, params, usys
    msg = (
        "No embed_point rule registered for "
        f"chart={type(intrinsic_chart)!r} into ambient={type(ambient_chart)!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def embed_point(
    intrinsic: cxc.SphericalTwoSphere,
    embedding: TwoSphereIn3D,
    p_pos: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    x_sph = embedding._embed_sph3d(intrinsic, p_pos, usys=usys)
    return cxc.point_realization_map(embedding.ambient, cxc.sph3d, x_sph, usys=usys)


# ===================================================================
# Point projection


@plum.dispatch
def project_point(
    embedded: EmbeddedChart,  # type: ignore[type-arg]
    p_ambient: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
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
        Ambient coordinates keyed by ``embedded.ambient.components`` with
        length units for ``Cart3D``.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CDict
        Intrinsic coordinates keyed by ``embedded.components``; for
        ``SphericalTwoSphere``, ``theta`` and ``phi`` are angles.

    Notes
    -----
    - Inputs are normalized by ``r`` so near-sphere points are accepted.
    - At the poles where ``sin(theta)=0``, ``phi`` is set to 0 by convention.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(1.0, "km"), ambient=cxc.cart3d)
    >>> chart = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)
    >>> p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = cxe.embed_point(chart, p)
    >>> p2 = cxe.project_point(chart, q)
    >>> bool(
    ...     jnp.allclose(
    ...         u.uconvert("rad", p2["theta"]).value,
    ...         u.uconvert("rad", p["theta"]).value,
    ...     )
    ... )
    True

    """
    return cxeapi.project_point(
        embedded.intrinsic, embedded.embedding, p_ambient, usys=usys
    )


@plum.dispatch
def project_point(
    embedded: EmbeddedManifold,
    p_ambient: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Project ambient coordinates onto intrinsic coordinates (manifold)."""
    return cxeapi.project_point(
        embedded.embedding.intrinsic, embedded.embedding, p_ambient, usys=usys
    )


@plum.dispatch
def project_point(
    intrinsic: cxc.AbstractChart,  # type: ignore[type-arg]
    ambient: cxc.AbstractChart,  # type: ignore[type-arg]
    p_ambient: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    del p_ambient, usys
    msg = (
        "No project_point rule registered for "
        f"chart={type(intrinsic)!r} from ambient={type(ambient)!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def project_point(
    intrinsic: cxc.SphericalTwoSphere,
    embedding: TwoSphereIn3D,
    p_ambient: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    x_sph = cxc.point_realization_map(
        cxc.sph3d, embedding.ambient, p_ambient, usys=usys
    )
    return embedding._project_sph3d(intrinsic, x_sph, usys=usys)


# ===================================================================
# Tangent embedding


@plum.dispatch
def embed_tangent(
    embedded: EmbeddedChart,  # type: ignore[type-arg]
    v_chart: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
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
    CDict
        Ambient physical tangent components keyed by
        ``embedded.ambient.components``.

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
    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(1.0, "km"), ambient=cxc.cart3d)
    >>> chart = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}
    >>> v_cart = cxe.embed_tangent(chart, v_tan, at=p)
    >>> bool(jnp.allclose(u.uconvert("km/s", v_cart["z"]).value, -1.0))
    True

    """
    return cxtapi.phys_tangent_basis_change(
        embedded.ambient, embedded, v_chart, at=at, usys=usys
    )


@plum.dispatch
def embed_tangent(
    embedded: EmbeddedManifold,
    v_chart: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    """Embed intrinsic physical tangent components into ambient (manifold)."""
    echart = EmbeddedChart(
        intrinsic=embedded.embedding.intrinsic, embedding=embedded.embedding
    )
    return cxtapi.phys_tangent_basis_change(
        echart.ambient, echart, v_chart, at=at, usys=usys
    )


# ===================================================================
# Tangent Projection


@plum.dispatch
def project_tangent(
    embedded: EmbeddedChart,  # type: ignore[type-arg]
    v_ambient: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
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
        keyed by ``embedded.ambient.components``.
    at
        Intrinsic point coordinates used to evaluate the tangent frame.
    usys
        Unit system for input quantities. Only needed if input is raw arrays,
        not {class}`~unxt.AbstractQuantity` objects.

    Returns
    -------
    CDict
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
    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(1.0, "km"), ambient=cxc.cart3d)
    >>> chart = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}
    >>> v_cart = cxe.embed_tangent(chart, v_tan, at=p)
    >>> v_back = cxe.project_tangent(chart, v_cart, at=p)
    >>> bool(jnp.allclose(u.uconvert("km/s", v_back["theta"]).value, 1.0))
    True

    """
    p_pos_ambient = cxeapi.embed_point(embedded, at, usys=usys)
    return cxtapi.phys_tangent_basis_change(
        embedded, embedded.ambient, v_ambient, at=p_pos_ambient, usys=usys
    )


@plum.dispatch
def project_tangent(
    embedded: EmbeddedManifold,
    v_ambient: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    """Project ambient physical components onto tangent space (manifold)."""
    echart = EmbeddedChart(
        intrinsic=embedded.embedding.intrinsic, embedding=embedded.embedding
    )
    p_pos_ambient = cxeapi.embed_point(echart, at, usys=usys)
    return cxtapi.phys_tangent_basis_change(
        echart, echart.ambient, v_ambient, at=p_pos_ambient, usys=usys
    )
