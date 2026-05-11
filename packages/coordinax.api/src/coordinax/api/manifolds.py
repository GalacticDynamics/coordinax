"""Vector API for coordinax."""

__all__ = (
    "guess_manifold",
    "scale_factors",
    "angle_between",
    "pt_embed",
    "pt_project",
    "pt_map",
)

from typing import TYPE_CHECKING, Any

import plum
import unxt as u

from ._custom_types import CDict

if TYPE_CHECKING:
    import coordinax.manifolds  # noqa: ICN001


@plum.dispatch.abstract
def guess_manifold(*args: Any, **kwargs: Any) -> "coordinax.manifolds.AbstractManifold":
    """Guess the manifold from arguments.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EuclideanManifold(2)
    >>> guess_manifold(M) is M
    True

    >>> cxm.guess_manifold({"x": 1, "y": 2, "z": 3})
    Rn(3)

    >>> cxm.guess_manifold(cxc.sph3d)
    Rn(3)

    >>> cxm.guess_manifold(cxc.sph2)
    HyperSphericalManifold(ndim=2)

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def scale_factors(chart: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Return the diagonal entries of the metric matrix.

    Dispatches on the first argument (metric or manifold) and the chart.
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def angle_between(
    metric_or_manifold: Any,
    chart: Any,
    uvec: Any,
    vvec: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    r"""Return the metric angle between two nonzero tangent vectors.

    The inputs ``uvec`` and ``vvec`` are component dictionaries representing
    tangent-vector components in the coordinate basis of ``chart``. The metric
    is evaluated at a base point supplied via ``at=...``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EuclideanManifold(2)
    >>> at = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> uvec = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
    >>> vvec = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m")}
    >>> cxm.angle_between(M, cxc.cart2d, uvec, vvec, at=at)
    Angle(1.57079633, 'rad')

    >>> metric = cxm.EuclideanMetric(3)
    >>> at_sph = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> u_tan = {"r": u.Q(0.0, "m"), "theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"r": u.Q(0.0, "m"), "theta": u.Angle(0.0, "rad"), "phi": u.Angle(1.0, "rad")}
    >>> cxm.angle_between(metric, cxc.sph3d, u_tan, v_tan, at=at_sph)
    Angle(1.57079633, 'rad')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pt_embed(
    p_pos: CDict, embedded: object, /, *, usys: u.AbstractUnitSystem | None = None
) -> CDict:
    r"""Embed intrinsic point coordinates into ambient coordinates.

    This function maps point coordinates from an intrinsic chart chart
    on an embedded manifold to the corresponding ambient space coordinates. It
    is the fundamental operation for working with embedded manifolds such as
    spheres, cylinders, or other submanifolds of Euclidean space.

    Mathematical Definition:

    Given an embedding $\iota: M \to \mathbb{R}^n$ of a manifold $M$ into
    ambient space $\mathbb{R}^n$, and intrinsic coordinates $q = (q^1, \ldots,
    q^k)$ on a chart $U \subset M$, this function computes the ambient
    coordinates:

    $$ x = \iota(q) = (x^1(q), \ldots, x^n(q)) $$

    For example, embedding the 2-sphere $S^2$ into $\mathbb{R}^3$ using
    spherical coordinates $(\theta, \phi)$:

    $$
    x &= R \sin\theta \cos\phi \\
    y &= R \sin\theta \sin\phi \\
    z &= R \cos\theta
    $$

    where $R$ is the radius parameter stored in the embedding object.

    Parameters
    ----------
    embedded
        The embedded manifold chart, typically an
        ``coordinax.manifolds.EmbeddedChart`` instance. This encapsulates:

        - ``intrinsic``: The intrinsic chart (e.g., ``SphericalTwoSphere``)
        - ``embedding``: An ``AbstractEmbedding`` that owns the ambient
          chart and any parameters (e.g., ``TwoSphereIn3D`` with ``radius``)
    p_pos
        Dictionary of intrinsic position coordinates. Keys must match
        ``embedded.components`` (e.g., ``"theta"`` and ``"phi"``
        for ``SphericalTwoSphere``). Values must have appropriate dimensions (e.g.,
        angles for angular coordinates).
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CDict
        Dictionary of ambient position coordinates. Keys match
        ``embedded.ambient.components`` (e.g., ``"x"``, ``"y"``, ``"z"``
        for ``Cart3D``). Values have dimensions appropriate for the ambient
        space (e.g., length for Cartesian coordinates).

    Raises
    ------
    NotImplementedError
        If no embedding rule is registered for the specific combination of
        intrinsic chart and embedding type.
    ValueError
        If required parameters are missing from the embedding or have
        incorrect dimensions.

    Notes
    -----
    - This is a point-only transformation. It does not handle velocities or
      other time derivatives. Use ``embed_tangent`` for differential quantities.

    - Embedding parameters (like ``radius`` for ``TwoSphereIn3D``) are stored
      on the embedding object and must have appropriate physical dimensions.

    - The embedding is purely geometric and does not encode physical components
      or metric information. Physical components require additional metric data.

    - Singularities of the intrinsic chart (e.g., poles on the sphere at
      $\theta = 0, \pi$) are inherited by the embedding but may not be
      problematic in the ambient space.

    See Also
    --------
    pt_project : Inverse operation projecting ambient to intrinsic
        coordinates
    embed_tangent : Embedding for tangent vector components
    coordinax.manifolds.EmbeddedChart : Container for embedded manifold
        charts

    Examples
    --------
    Embedding a point on the 2-sphere into 3D Cartesian coordinates:

    >>> import quaxed.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> import wadler_lindig as wl

    Create an embedded manifold for a sphere of radius 5 km:

    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(5, "km")))

    Embed a point at the equator:

    >>> p_intrinsic = {
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),  # equator
    ...     "phi": u.Angle(0.0, "rad"),           # along x-axis
    ... }
    >>> p_ambient = cxm.pt_embed(p_intrinsic, chart)
    >>> wl.pprint(p_ambient, short_arrays='compact', named_unit=False)
    {'r': Quantity(5, 'km'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    Verify the point lies on the sphere:

    >>> p_ambient_cart = cxm.pt_map(p_ambient, chart.ambient, cxc.cart3d)
    >>> r2 = jnp.linalg.norm(jnp.array(list(p_ambient_cart.values())))
    >>> jnp.allclose(r2, u.Q(25.0, "km"), atol=u.Q(1e-10, "km"))
    Array(False, dtype=bool)

    Embedding a point at the north pole:

    >>> p_pole = {
    ...     "theta": u.Angle(0.0, "rad"),  # north pole
    ...     "phi": u.Angle(0.0, "rad"),    # phi is arbitrary at poles
    ... }
    >>> p_ambient_pole = cxm.pt_embed(p_pole, chart)
    >>> wl.pprint(p_ambient_pole, short_arrays='compact', named_unit=False)
    {'r': Quantity(5, 'km'), 'theta': Angle(0., 'rad'), 'phi': Angle(0., 'rad')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pt_project(*args: object, usys: u.AbstractUnitSystem | None = None) -> CDict:
    r"""Project ambient space coordinates onto intrinsic chart coordinates.

    This function performs the inverse operation of ``pt_embed``, taking coordinates
    from the ambient (embedding) space and projecting them back onto the intrinsic
    manifold chart. It's essential for converting between extrinsic and intrinsic
    charts of points on embedded submanifolds.

    Mathematical Definition:

    For an embedding $\iota: M \to \mathbb{R}^n$ of a $k$-dimensional manifold
    $M$ into $n$-dimensional Euclidean space, the projection $\pi: \mathbb{R}^n
    \to M$ (or more precisely, onto a chart $U \subset M$) satisfies:

    $$ \pi \circ \iota = \mathrm{id}_M $$

    meaning that projecting after embedding returns the original point (up to
    numerical precision and chart domain).

    For a 2-sphere $S^2 \subset \mathbb{R}^3$ with radius $R$, given Cartesian
    coordinates $(x, y, z)$:

    $$
        r &= \sqrt{x^2 + y^2 + z^2} \\
        \theta &= \arccos\!\left(\frac{z}{r}\right) \in [0, \pi] \\
        \phi &= \operatorname{atan2}(y, x) \in (-\pi, \pi]
    $$

    The projection normalizes the input by $r$, so it maps any point in
    $\mathbb{R}^3 \setminus \{0\}$ to the sphere.

    **Key properties**:

    - **Local inverse**: On the manifold, $\pi(\iota(q)) = q$ exactly
    - **Normalization**: Points near the manifold are projected onto it (e.g.,
      points near the sphere are normalized to lie exactly on it)
    - **Singularities**: Projection may have singularities where the manifold's
      chart has coordinate singularities (e.g., poles of a sphere)
    - **Not globally defined**: Projection is typically only defined for points
      in a neighborhood of the manifold, not all of $\mathbb{R}^n$

    Parameters
    ----------
    *args
        Typically an ``EmbeddedChart`` chart and a dictionary of
        ambient coordinates, but the function supports multiple dispatch patterns.
        Common signature:

        - ``embedded`` : EmbeddedChart
            The embedded manifold chart specifying the chart and ambient
            space
        - ``p_ambient`` : CDict
            Ambient coordinates keyed by ``embedded.ambient.components``
            (e.g., ``{"x": ..., "y": ..., "z": ...}`` for Cartesian ambient
            space)
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CDict
        Intrinsic chart coordinates keyed by the manifold chart's components.
        For a sphere, these are typically ``{"theta": ..., "phi": ...}``.

    Raises
    ------
    NotImplementedError
        If no projection is defined for the given manifold and ambient space.
    ValueError
        If the ambient point cannot be projected (e.g., origin for sphere).

    Notes
    -----
    - **Normalization**: Implementations often normalize inputs to handle points
      that are close to but not exactly on the manifold. For a sphere, any
      non-zero point in $\mathbb{R}^3$ is normalized to radius $R$.

    - **Coordinate singularities**: At chart singularities (e.g., sphere poles
      where $\sin\theta = 0$), convention determines the value of singular
      coordinates (typically $\phi = 0$ at poles).

    - **Round-trip accuracy**: For points on the manifold,
      ``pt_project(pt_embed(q, embedded), embedded)`` should return ``q`` up to
      floating-point precision.

    - **Extension to nearby points**: Projection extends the manifold chart to a
      neighborhood in the ambient space, useful for perturbed or approximate data.

    - **Orthogonal projection**: For Riemannian manifolds with the induced metric,
      this is often the orthogonal projection onto the manifold (shortest distance
      from the ambient point to the manifold).

    See Also
    --------
    pt_embed : Embed intrinsic chart coordinates into ambient space
    project_tangent : Project ambient velocity/acceleration onto tangent space

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> import wadler_lindig as wl

    Project Cartesian coordinates onto a 2-sphere.
    Create an embedded 2-sphere of radius 1 km:

    >>> sphere = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(1, "km")))

    Project a point in 3D Cartesian space onto the sphere.
    Start with a point slightly off the sphere:

    >>> p_cart = {"x": u.Q(0.5, "km"), "y": u.Q(0.5, "km"), "z": u.Q(0.707, "km")}
    >>> p_sphere = cxm.pt_project(p_cart, cxc.cart3d, sphere)
    >>> wl.pprint(p_sphere, short_arrays='compact', named_unit=False)
    {'theta': Quantity(0.78547367, 'rad'), 'phi': Quantity(0.78539816, 'rad')}

    The projection normalizes the point to lie exactly on the sphere.  Verify
    round-trip accuracy (project after embed returns original point):

    >>> q_sphere = {"theta": u.Q(jnp.pi / 3, "rad"),
    ...             "phi": u.Q(jnp.pi / 4, "rad")}
    >>> q_cart = cxm.pt_embed(q_sphere, sphere)
    >>> q_recovered = cxm.pt_project(q_cart, sphere)
    >>> all(jax.tree.map(jnp.isclose, q_sphere, q_recovered))
    True

    Project from an arbitrary point in space (not on sphere).
    The projection normalizes the radius:

    >>> p_far = {"x": u.Q(2.0,"km"), "y": u.Q(2.0,"km"), "z": u.Q(2.0,"km")}
    >>> p_normalized = cxm.pt_project(p_far, cxc.cart3d, sphere)
    >>> # Direction is preserved: all coordinates equal → theta ≈ 54.7°, phi = 45°
    >>> wl.pprint(p_normalized, short_arrays='compact', named_unit=False)
    {'theta': Quantity(0.95531662, 'rad'), 'phi': Quantity(0.78539816, 'rad')}

    Handle coordinate singularities at the poles.
    At the north pole ($\theta = 0$), $\phi$ is conventionally set to 0:

    >>> p_north = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"),
    ...            "z": u.Q(1.0, "km")}  # North pole
    >>> cxm.pt_project(p_north, cxc.cart3d, sphere)
    {'theta': Q(0., 'rad'), 'phi': Q(0., 'rad')}

    At the south pole ($\theta = \pi$), $\phi$ is also set to 0:

    >>> p_south = { "x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(-1, "km")}
    >>> cxm.pt_project(p_south, cxc.cart3d, sphere)
    {'theta': Q(3.14159265, 'rad'), 'phi': Q(0., 'rad')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pt_map(*args: Any, **kwargs: Any) -> CDict:
    r"""Transform position coordinates from one chart to another.

    This function implements the most general point-coordinate map between two
    compatible chart representations of the same geometric point. It is a
    point-wise map that preserves the physical location while changing the
    coordinate description.

    For charts in the same atlas on the same manifold, this reduces to the
    ordinary chart transition map handled by `pt_map`. It is the
    intrinsic coordinate-change operation: the underlying point on the manifold
    is unchanged, and only its coordinate representation is changed.

    However, this function is not restricted to two charts on the same manifold.
    It may also represent a **realization-style** map between charts attached to
    different manifolds when one is a realization of the other, such as an
    intrinsic chart on an embedded manifold and a chart on its ambient manifold.
    In that case, this function may change both the chart and the manifold in
    which the point is being represented.

    Mathematical Definition:

    Let $(U, \varphi_{\mathrm{from}})$ and $(V, \varphi_{\mathrm{to}})$ be
    charts on the same manifold $M$, with overlapping domains. The transition
    map is

    $$
        \varphi_{\mathrm{to}} \circ \varphi_{\mathrm{from}}^{-1}
        :
        \varphi_{\mathrm{from}}(U \cap V)
        \to
        \varphi_{\mathrm{to}}(U \cap V).
    $$

    If a point $p \in U \cap V$ has coordinates
    $q = \varphi_{\mathrm{from}}(p)$, then this function returns
    $p' = \varphi_{\mathrm{to}}(p)$ for the same manifold point.

    More generally, if $\varphi_{\mathrm{from}} : U \subset M \to \mathbb{R}^n$
    and $\psi_{\mathrm{to}} : W \subset N \to \mathbb{R}^m$ are chart maps on
    manifolds $M$ and $N$, and there is a point map $F : M \supset U \to W
    \subset N$, then `pt_map` represents the coordinate expression

    $$ \psi_{\mathrm{to}} \circ F \circ \varphi_{\mathrm{from}}^{-1}. $$

    - **3D Spherical → Cartesian**:

      $$
          x &= r \sin\theta \cos\phi \\ y &= r \sin\theta \sin\phi \\ z &= r
          \cos\theta
      $$

    Raises
    ------
    NotImplementedError
        If no transformation rule is registered for the specific pair of charts
        ``(to_chart, from_chart)``.

    Notes
    -----
    - This is a **position-only** transformation.
    - This function may map between charts on the same manifold or across
      manifolds, provided a compatible point map is defined between them.
    - Transformations preserve physical dimensions. For example, converting from
      polar to Cartesian preserves that ``r`` has length dimension and produces
      ``x`` and ``y`` with length dimension.
    - Some transformations may introduce singularities (e.g., polar coordinates
      at the origin, spherical coordinates at poles).
    - Transformations are composable: transforming $A \to B \to C$ yields the
      same result as a direct $A \to C$ transformation (up to numerical
      precision).
    - Identity transformations (same ``from_chart`` and ``to_chart``) return the
      input unchanged.

    See Also
    --------
    pt_map : transform position coordinates between charts on the
    same manifold.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Transform from 2D polar to Cartesian:

    >>> p_polar = {"r": u.Q(2.0, "m"), "theta": u.Angle(jnp.pi / 4, "rad")}
    >>> cxc.pt_map(p_polar, cxc.polar2d, cxc.cart2d)
    {'x': Q(1.41421356, 'm'), 'y': Q(1.41421356, 'm')}

    Transform from 3D spherical to Cartesian:

    >>> p_sph = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad"),
    ...          "r": u.Q(5.0, "km")}
    >>> cxc.pt_map(p_sph, cxc.sph3d, cxc.cart3d)
    {'x': Q(5., 'km'), 'y': Q(0., 'km'), 'z': Q(3.061617e-16, 'km')}

    Transform from Cartesian to cylindrical:

    >>> p_xyz = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxc.pt_map(p_xyz, cxc.cart3d, cxc.cyl3d)
    {'rho': Q(5., 'm'), 'phi': Q(0.92729522, 'rad'), 'z': Q(5., 'm')}

    """
    del args, kwargs  # Unused in abstract method
    raise NotImplementedError  # pragma: no cover
