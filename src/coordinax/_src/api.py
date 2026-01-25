"""Internal API."""

__all__ = (
    # Charts
    "cartesian_chart",
    "guess_chart",
    # Embeddings
    "embed_point",
    "project_point",
    "embed_tangent",
    "project_tangent",
    # Metrics
    "physicalize",
    "coordinateize",
    "lower_index",
    "raise_index",
    "metric_of",
    # Reference frames
    "frame_of",
    "frame_transform_op",
    # Roles
    "as_pos",
    "guess_role",
    # Transformations
    "point_transform",
    "physical_tangent_transform",
    "coord_transform",
    "cotangent_transform",
    "frame_cart",
    "pushforward",
    "pullback",
    # Operators
    "apply_op",
    "simplify",
)

from jaxtyping import Array
from typing import TYPE_CHECKING, Any

import plum

from .custom_types import CsDict, OptUSys

if TYPE_CHECKING:
    import coordinax.charts  # noqa: ICN001
    import coordinax.metrics  # noqa: ICN001
    import coordinax.roles  # noqa: ICN001


# ===================================================================
# Chart


@plum.dispatch.abstract
def cartesian_chart(obj: Any, /) -> "coordinax.charts.AbstractChart":  # type: ignore[type-arg]
    r"""Return the corresponding Cartesian chart for a given chart.

    This function provides the canonical flat-space Cartesian chart
    associated with any coordinate system. It maps each chart to its
    natural Cartesian equivalent in the same dimensional space.

    Mathematical Definition
    -----------------------
    For a coordinate chart $\mathcal{R}$ in $n$-dimensional
    space, this returns the Cartesian chart $\mathcal{C}_n$ such that:

    $$
    \mathrm{cartesian\_chart}(\mathcal{R}) = \mathcal{C}_n
    $$

    where $\mathcal{C}_n \in \{\text{Cart1D}, \text{Cart2D}, \text{Cart3D},
    \text{CartND}\}$ depending on $n = \text{ndim}$.

    The Cartesian chart uses orthonormal basis vectors with components
    typically denoted $(x)$, $(x, y)$, $(x, y, z)$, or
    $(q_1, \ldots, q_n)$ for arbitrary dimension.

    Parameters
    ----------
    obj : Any
        A coordinate chart instance (e.g., `coordinax.charts.sph3d`,
        `coordinax.charts.polar2d`) or any object for which a Cartesian
        equivalent is defined.

    Returns
    -------
    coordinax.charts.AbstractChart
        The Cartesian chart in the same dimensional space:

        - 1D charts → ``Cart1D`` (component: ``x``)
        - 2D charts → ``Cart2D`` (components: ``x``, ``y``)
        - 3D charts → ``Cart3D`` (components: ``x``, ``y``, ``z``)
        - N-D charts → ``CartND`` (components: ``q``)

    Raises
    ------
    NotImplementedError
        If no Cartesian chart is defined for the input object.

    Notes
    -----
    - Cartesian charts use the Euclidean metric with orthonormal bases.
    - This function does **not** perform coordinate transformation; it only
      returns the chart type. Use ``point_transform`` for actual
      coordinate conversion.
    - All standard Euclidean coordinate systems (spherical, cylindrical, polar)
      map to their dimensional Cartesian equivalent.
    - For embedded manifolds, this returns the Cartesian form of the ambient
      space, not the intrinsic coordinates.

    See Also
    --------
    point_transform : Transform coordinates between charts
    coordinax.charts.AbstractChart : Base class for coordinate charts

    Examples
    --------
    >>> import coordinax as cx

    1D coordinate systems map to Cart1D:

    >>> cx.charts.cartesian_chart(cx.charts.cart1d)
    Cart1D(...)

    >>> cx.charts.cartesian_chart(cx.charts.radial1d)
    Cart1D(...)

    2D coordinate systems map to Cart2D:

    >>> cx.charts.cartesian_chart(cx.charts.cart2d)
    Cart2D(...)

    >>> cx.charts.cartesian_chart(cx.charts.polar2d)
    Cart2D(...)

    3D coordinate systems map to Cart3D:

    >>> cx.charts.cartesian_chart(cx.charts.cart3d)
    Cart3D(...)

    >>> cx.charts.cartesian_chart(cx.charts.sph3d)
    Cart3D(...)

    >>> cx.charts.cartesian_chart(cx.charts.cyl3d)
    Cart3D(...)

    N-dimensional systems map to CartND:

    >>> cx.charts.cartesian_chart(cx.charts.cartnd)
    CartND(...)

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_chart(*args: Any) -> "coordinax.charts.AbstractChart":  # type: ignore[type-arg]
    """Infer a Cartesian chart from the shape of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cx.charts.guess_chart(q)
    Cart3D()

    """
    raise NotImplementedError  # pragma: no cover


# ===================================================================
# Embedding


@plum.dispatch.abstract
def embed_point(
    embedded: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    p_pos: CsDict,
    /,
    *,
    usys: OptUSys = None,
) -> CsDict:
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

    where $R$ is the radius parameter stored in ``embedded.params["R"]``.

    Parameters
    ----------
    embedded
        The embedded manifold chart, typically an
        ``coordinax.embeddings.EmbeddedManifold`` instance. This encapsulates:

        - ``intrinsic_chart``: The intrinsic chart chart (e.g.,
          ``TwoSphere``)
        - ``ambient_chart``: The ambient space chart (e.g., ``Cart3D``)
        - ``params``: Embedding-specific parameters (e.g., ``{"R":
          Quantity(...)}`` for the sphere radius)
    p_pos
        Dictionary of intrinsic position coordinates. Keys must match
        ``embedded.intrinsic_chart.components`` (e.g., ``"theta"`` and ``"phi"``
        for ``TwoSphere``). Values must have appropriate dimensions (e.g.,
        angles for angular coordinates).
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
        Dictionary of ambient position coordinates. Keys match
        ``embedded.ambient_chart.components`` (e.g., ``"x"``, ``"y"``, ``"z"``
        for ``Cart3D``). Values have dimensions appropriate for the ambient
        space (e.g., length for Cartesian coordinates).

    Raises
    ------
    NotImplementedError
        If no embedding rule is registered for the specific combination of
        ``intrinsic_chart`` and ``ambient_chart``.
    ValueError
        If required parameters are missing from ``embedded.params`` or have
        incorrect dimensions.

    Notes
    -----
    - This is a point-only transformation. It does not handle velocities or
      other time derivatives. Use ``embed_tangent`` for differential quantities.

    - Embedding parameters (like radius ``R`` for spheres) must be provided in
      ``embedded.params`` and must have appropriate physical dimensions.

    - The embedding is purely geometric and does not encode physical components
      or metric information. Physical components require additional metric data.

    - Singularities of the intrinsic chart (e.g., poles on the sphere at
      $\theta = 0, \pi$) are inherited by the embedding but may not be
      problematic in the ambient space.

    See Also
    --------
    project_point : Inverse operation projecting ambient to intrinsic
        coordinates
    embed_tangent : Embedding for tangent vector components
    coordinax.embeddings.EmbeddedManifold : Container for embedded manifold
        charts

    Examples
    --------
    Embedding a point on the 2-sphere into 3D Cartesian coordinates:

    >>> import quaxed.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> import wadler_lindig as wl

    Create an embedded manifold for a sphere of radius 5 km:

    >>> embedding_chart = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                                        params={"R": u.Q(5.0, "km")})

    Embed a point at the equator:

    >>> p_intrinsic = {
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),  # equator
    ...     "phi": u.Angle(0.0, "rad"),           # along x-axis
    ... }
    >>> p_ambient = cxe.embed_point(embedding_chart, p_intrinsic)
    >>> wl.pprint(p_ambient, short_arrays='compact', named_unit=False)
    {'x': Quantity(5., 'km'), 'y': Quantity(0., 'km'), 'z': Quantity(3...-16, 'km')}

    Verify the point lies on the sphere:

    >>> r_squared = jnp.linalg.norm(jnp.array(list(p_ambient.values())))
    >>> jnp.allclose(r_squared, u.Q(25.0, "km"), atol=u.Q(1e-10, "km"))
    Array(False, dtype=bool)

    Embedding a point at the north pole:

    >>> p_pole = {
    ...     "theta": u.Angle(0.0, "rad"),  # north pole
    ...     "phi": u.Angle(0.0, "rad"),    # phi is arbitrary at poles
    ... }
    >>> p_ambient_pole = cxe.embed_point(embedding_chart, p_pole)
    >>> wl.pprint(p_ambient_pole, short_arrays='compact', named_unit=False)
    {'x': Quantity(0., 'km'), 'y': Quantity(0., 'km'), 'z': Quantity(5., 'km')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def project_point(*args: Any, usys: OptUSys = None) -> CsDict:
    r"""Project ambient space coordinates onto intrinsic chart coordinates.

    This function performs the inverse operation of ``embed_point``, taking coordinates
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
        Typically an ``EmbeddedManifold`` chart and a dictionary of
        ambient coordinates, but the function supports multiple dispatch patterns.
        Common signature:

        - ``embedded`` : EmbeddedManifold
            The embedded manifold chart specifying the chart and ambient
            space
        - ``p_ambient`` : CsDict
            Ambient coordinates keyed by ``embedded.ambient_chart.components``
            (e.g., ``{"x": ..., "y": ..., "z": ...}`` for Cartesian ambient
            space)
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
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
      ``project_point(embedded, embed_point(embedded, q))`` should return ``q`` up to
      floating-point precision.

    - **Extension to nearby points**: Projection extends the manifold chart to a
      neighborhood in the ambient space, useful for perturbed or approximate data.

    - **Orthogonal projection**: For Riemannian manifolds with the induced metric,
      this is often the orthogonal projection onto the manifold (shortest distance
      from the ambient point to the manifold).

    See Also
    --------
    embed_point : Embed intrinsic chart coordinates into ambient space
    project_tangent : Project ambient velocity/acceleration onto tangent space
    physical_tangent_transform : Transform vectors between charts

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> import wadler_lindig as wl

    Project Cartesian coordinates onto a 2-sphere.
    Create an embedded 2-sphere of radius 1 km:

    >>> sphere = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                               params={"R": u.Q(1.0, "km")})

    Project a point in 3D Cartesian space onto the sphere.
    Start with a point slightly off the sphere:

    >>> p_cart = {
    ...     "x": u.Q(0.5, "km"),   # sqrt(0.5^2 + 0.5^2 + 0.707^2) ≈ 1.0
    ...     "y": u.Q(0.5, "km"),
    ...     "z": u.Q(0.707, "km"),
    ... }
    >>> p_sphere = cxe.project_point(sphere, p_cart)
    >>> wl.pprint(p_sphere, short_arrays='compact', named_unit=False)
    {'theta': Quantity(0.78547367, 'rad'), 'phi': Quantity(0.78539816, 'rad')}

    The projection normalizes the point to lie exactly on the sphere.  Verify
    round-trip accuracy (project after embed returns original point):

    >>> q_sphere = {"theta": u.Q(jnp.pi / 3, "rad"),
    ...             "phi": u.Q(jnp.pi / 4, "rad")}
    >>> q_cart = cxe.embed_point(sphere, q_sphere)
    >>> q_recovered = cxe.project_point(sphere, q_cart)
    >>> all(jax.tree.map(jnp.isclose, q_sphere, q_recovered))
    True

    Project from an arbitrary point in space (not on sphere).
    The projection normalizes the radius:

    >>> p_far = {"x": u.Q(2.0,"km"), "y": u.Q(2.0,"km"), "z": u.Q(2.0,"km")}
    >>> p_normalized = cxe.project_point(sphere, p_far)
    >>> # Direction is preserved: all coordinates equal → theta ≈ 54.7°, phi = 45°
    >>> wl.pprint(p_normalized, short_arrays='compact', named_unit=False)
    {'theta': Quantity(0.95531662, 'rad'), 'phi': Quantity(0.78539816, 'rad')}

    Handle coordinate singularities at the poles.
    At the north pole ($\theta = 0$), $\phi$ is conventionally set to 0:

    >>> p_north = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"),
    ...            "z": u.Q(1.0, "km")}  # North pole
    >>> cxe.project_point(sphere, p_north)
    {'theta': Quantity(Array(0., dtype=float64, ...), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    At the south pole ($\theta = \pi$), $\phi$ is also set to 0:

    >>> p_south = { "x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(-1.0, "km")
    ... }
    >>> cx.embeddings.project_point(sphere, p_south)
    {'theta': Quantity(Array(3.14159265, dtype=float64, ...), unit='rad'),
    'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def embed_tangent(
    embedded: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    v_chart: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    r"""Embed intrinsic physical tangent components into ambient physical components.

    This function maps physical tangent vector components from an intrinsic chart
    chart on an embedded manifold to the corresponding ambient space
    components. It is essential for transforming velocities, accelerations, and
    other tangent vectors between intrinsic and ambient charts.

    Mathematical Definition:

    Given an embedding $\iota: M \to \mathbb{R}^n$ and a tangent vector
    $v \in T_q M$ at point $q$, the embedding of tangent vectors is
    defined via the pushforward (differential) of $\iota$:

    $$
        (\iota_*)_q: T_q M \to T_{\iota(q)} \mathbb{R}^n
    $$
    In coordinates, if $v = v^i \hat{e}_i(q)$ where $\{\hat{e}_i(q)\}$
    is an orthonormal frame on $M$ at $q$, then:

    $$
        v_{\text{ambient}} = B(q) \, v_{\text{chart}}
    $$
    where $B(q) = [\hat{e}_1(q) \cdots \hat{e}_k(q)]$ is the matrix of
    orthonormal basis vectors expressed in ambient coordinates.

    For a 2-sphere $S^2 \subset \mathbb{R}^3$ with
    $v = v^\theta \hat{e}_\theta + v^\phi \hat{e}_\phi$:

    $$
        \hat{e}_\theta &= (\cos\theta\cos\phi, \cos\theta\sin\phi, -\sin\theta) \\
        \hat{e}_\phi &= (-\sin\phi, \cos\phi, 0)
    $$
    **Key properties**:

    - **Uniform units**: All components of $v_{\text{chart}}$ must have the
      same physical dimension (e.g., all speed or all acceleration).
    - **Orthonormal frames**: The basis $\{\hat{e}_i\}$ is orthonormal with
      respect to the ambient Euclidean metric.
    - **Not coordinate differentials**: These are physical tangent components, not
      coordinate derivatives $dq^i$.

    Parameters
    ----------
    embedded : coordinax.charts.AbstractChart
        The embedded manifold chart, typically an
        {class}`coordinax.embeddings.EmbeddedManifold` instance specifying the
        intrinsic chart and ambient space.
    v_chart : CsDict
        Physical tangent vector components keyed by ``embedded.components``.
        All values must have uniform physical dimension (e.g., ``"km/s"`` for
        velocity, ``"km/s^2"`` for acceleration). These are **not** coordinate
        derivatives.
    at : CsDict
        Intrinsic point coordinates where the tangent frame is evaluated.
        Required because the orthonormal frame depends on the base point.  Keys
        must match ``embedded.components``.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
        Ambient physical tangent components keyed by
        ``embedded.ambient_chart.components``. These have the same physical
        dimension as the input components.

    Raises
    ------
    NotImplementedError
        If no embedding rule is registered for the specific manifold.
    ValueError
        If components have inconsistent units or missing base point.

    Notes
    -----
    - **Physical components, not differentials**: The input and output are
      physical tangent vector components with uniform units, not coordinate
      differentials $dq^i$ which would have mixed dimensions.

    - **Base point required**: The ``at`` parameter is mandatory because the
      tangent frame basis $\{\hat{e}_i(q)\}$ depends on the point $q$.

    - **Orthonormality**: The ambient metric (Euclidean) is used to define
      orthonormality of the basis vectors.

    - **Linear operation**: This is a linear map on each tangent space, but the
      matrix $B(q)$ varies with the base point.

    See Also
    --------
    project_tangent :
        Inverse operation projecting ambient to intrinsic components
    embed_point : Embed point coordinates (not tangent vectors)

    Examples
    --------
    Embedding a tangent vector on the 2-sphere:

    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u

    Create an embedded 2-sphere:

    >>> sphere = cx.charts.EmbeddedManifold(
    ...     intrinsic_chart=cx.charts.twosphere,
    ...     ambient_chart=cx.charts.cart3d,
    ...     params={"R": u.Q(1.0, "km")},
    ... )

    Define a point and a tangent vector (velocity) at that point:

    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_chart = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}

    Embed the tangent vector to ambient Cartesian components:

    >>> v_ambient = cx.embeddings.embed_tangent(sphere, v_chart, at=p)
    >>> jnp.allclose(v_ambient["x"].value, 0.0, atol=1e-10)
    Array(True, dtype=bool)
    >>> jnp.allclose(v_ambient["y"].value, 0.0, atol=1e-10)
    Array(True, dtype=bool)
    >>> jnp.allclose(v_ambient["z"].value, -1.0, atol=1e-10)
    Array(True, dtype=bool)

    At the equator moving in the $\theta$ direction (southward),
    the ambient velocity points in the $-z$ direction.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def project_tangent(
    embedded: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    v_ambient: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    r"""Project ambient physical components onto the manifold tangent space.

    This function performs the inverse operation of ``embed_tangent``, projecting
    physical tangent vector components from the ambient space onto the intrinsic
    manifold chart. This is the pullback operation on tangent vectors.

    Mathematical Definition:

    For an embedding $\iota: M \to \mathbb{R}^n$, the projection is the pullback
    (or restriction) of tangent vectors:

    $$
        (\iota^*)_q: T_{\iota(q)} \mathbb{R}^n \to T_q M
    $$
    In terms of the orthonormal frame $B(q) = [\hat{e}_1(q) \cdots \hat{e}_k(q)]$:

    $$
        v_{\text{chart}} = B(q)^{\mathsf{T}} v_{\text{ambient}}
    $$
    For a Euclidean ambient space, this is an **orthogonal projection** onto the
    tangent space of the manifold. The component of $v_{\text{ambient}}$ normal
    to the manifold is discarded.

    **Key properties**:

    - **Orthogonal projection**: Uses the ambient Euclidean metric
    - **Loses normal component**: Only the tangential part is retained
    - **Left inverse**: $\text{project}(\text{embed}(v, at=q), at=q) = v$
    - **Not globally defined**: Only defined for points where the tangent space exists

    Parameters
    ----------
    embedded
        The embedded manifold chart specifying the intrinsic chart
        and ambient space.
    v_ambient
        Ambient physical tangent components keyed by
        ``embedded.ambient_chart.components``. All components must have uniform
        physical dimension (e.g., all speed or all acceleration). These are
        physical components, **not** coordinate derivatives.
    at
        Intrinsic point coordinates where the tangent frame is evaluated.
        Required because the projection depends on the base point.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
        Intrinsic tangent components keyed by ``embedded.components``.
        These have the same physical dimension as the input.

    Raises
    ------
    NotImplementedError
        If no projection is defined for the given manifold.
    ValueError
        If missing base point or inconsistent component units.

    Notes
    -----
    - **Physical tangent components**: Input and output are physical vector
      components with uniform units, not coordinate differentials.

    - **Orthogonal projection**: In Euclidean ambient space, this is the
      orthogonal projection onto $T_q M$. The normal component is
      discarded.

    - **Base point required**: The ``at`` parameter is mandatory because the
      tangent space depends on the point.

    - **Round-trip property**: For vectors in the tangent space,
      ``project_tangent(embed_tangent(v, at=q), at=q)`` returns ``v``.

    - **Not surjective from ambient**: Only ambient vectors with zero normal
      component can be exactly chartresented in the intrinsic chart.

    See Also
    --------
    embed_tangent : Embed intrinsic tangent components to ambient space
    project_point : Project point coordinates (not tangent vectors)
    physical_tangent_transform : General tangent vector transformation

    Examples
    --------
    Project an ambient velocity onto the 2-sphere tangent space:

    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u

    Create an embedded 2-sphere:

    >>> sphere = cx.charts.EmbeddedManifold(
    ...     intrinsic_chart=cx.charts.twosphere,
    ...     ambient_chart=cx.charts.cart3d,
    ...     params={"R": u.Q(1.0, "km")},
    ... )

    Define a point and an ambient velocity:

    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_ambient = {
    ...     "x": u.Q(0.0, "km/s"),
    ...     "y": u.Q(1.0, "km/s"),  # tangent to sphere
    ...     "z": u.Q(0.5, "km/s"),  # has normal component
    ... }

    Project onto the tangent space:

    >>> v_chart = cx.embeddings.project_tangent(sphere, v_ambient, at=p)
    >>> jnp.allclose(v_chart["theta"].value, -0.5, atol=1e-10)
    Array(True, dtype=bool)
    >>> jnp.allclose(v_chart["phi"].value, 1.0, atol=1e-10)
    Array(True, dtype=bool)

    Note that the normal component (radial $z$) is discarded.

    Verify round-trip property:

    >>> v_chart_orig = {"theta": u.Q(2.0, "km/s"), "phi": u.Q(3.0, "km/s")}
    >>> v_amb = cx.embeddings.embed_tangent(sphere, v_chart_orig, at=p)
    >>> v_recovered = cx.embeddings.project_tangent(sphere, v_amb, at=p)
    >>> jnp.allclose(
    ...     u.uconvert("km/s", v_recovered["theta"]).value,
    ...     u.uconvert("km/s", v_chart_orig["theta"]).value,
    ... )
    Array(True, dtype=bool)

    """
    raise NotImplementedError  # pragma: no cover


# ======================================================================
# Metrics


@plum.dispatch.abstract
def metric_of(*args: Any) -> "coordinax.metricsAbstractMetric":
    r"""Return the metric tensor associated with a coordinate chart.

    The metric tensor encodes the geometry of space in a given coordinate
    system, defining how to measure distances, angles, and volumes. It is
    essential for computing orthonormal frames, transforming vectors, and
    determining the intrinsic curvature properties of the space.

    Mathematical Definition:

    For a coordinate chart with coordinates $q = (q^1, \ldots, q^n)$,
    the metric tensor $g$ is a symmetric, positive-definite matrix field:

    $$ g_{ij}(q) = \mathbf{e}_i(q) \cdot \mathbf{e}_j(q) $$

    where $\mathbf{e}_i = \partial \mathbf{x} / \partial q^i$ are the coordinate
    basis vectors. The metric determines the infinitesimal line element:

    $$ ds^2 = g_{ij} \, dq^i \, dq^j $$

    Common examples:

    - **Cartesian** (Euclidean):

      $$
          g_{ij} = \delta_{ij} = \begin{pmatrix}
            1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1
          \end{pmatrix}
      $$
    - **2D Polar**:

      $$
          g = \begin{pmatrix}
            1 & 0 \\ 0 & r^2
          \end{pmatrix}, \quad ds^2 = dr^2 + r^2 d\theta^2
      $$
    - **3D Spherical**:

      $$
          g = \begin{pmatrix}
            1 & 0 & 0 \\ 0 & r^2 & 0 \\ 0 & 0 & r^2 \sin^2\theta
          \end{pmatrix}
      $$
    - **Minkowski** (spacetime):

      $$ g = \mathrm{diag}(-c^2, 1, 1, 1) $$

    Parameters
    ----------
    *args
        Typically a single coordinate chart instance (e.g.,
        ``cx.charts.sph3d``, ``cx.charts.cart3d``), but the function supports
        multiple dispatch patterns for more complex scenarios.

    Returns
    -------
    coordinax.metricsAbstractMetric
        The metric tensor associated with the chart. Common types
        include:

        - ``EuclideanMetric``: Flat space with $g_{ij} = \delta_{ij}$
        - ``MinkowskiMetric``: Flat spacetime with signature $(-,+,+,+)$
        - ``SphereMetric``: Riemannian metric on a sphere
        - Custom metric types for specialized coordinate systems

    Raises
    ------
    NotImplementedError
        If no metric is defined for the given arguments.

    Notes
    -----
    - The metric determines whether a space is **Euclidean** (flat),
      **Riemannian** (curved spatial manifold), or **pseudo-Riemannian**
      (spacetime with mixed signature).

    - **Orthonormal frames** are computed from the metric via Gram-Schmidt or
      Cholesky decomposition of $g_{ij}$.

    - The metric is used in ``tangent_transform`` to correctly handle vector
      transformations in curvilinear coordinates.

    - For embedded manifolds (e.g., sphere in 3D), the metric is the induced
      (pullback) metric from the ambient space.

    - Position-dependent: Most metrics vary with position (e.g., $r^2$ in polar
      coordinates), requiring ``p_pos`` when computing the metric matrix.

    See Also
    --------
    physical_tangent_transform : Uses the metric for vector transformations
    frame_cart : Constructs orthonormal frames from the metric
    coordinax.metricsAbstractMetric : Base class for metric implementations

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Get the metric for 3D Cartesian coordinates (Euclidean):

    >>> metric_cart = cx.metrics.metric_of(cx.charts.cart3d)
    >>> isinstance(metric_cart, cx.metrics.EuclideanMetric)
    True

    The metric matrix is the identity:

    >>> g_cart = metric_cart.metric_matrix(cx.charts.cart3d, {})
    >>> g_cart.shape
    (3, 3)
    >>> import jax.numpy as jnp
    >>> jnp.allclose(g_cart, jnp.eye(3))
    Array(True, dtype=bool)

    Get the metric for 3D spherical coordinates:

    >>> metric_sph = cx.metrics.metric_of(cx.charts.sph3d)
    >>> isinstance(metric_sph, cx.metrics.EuclideanMetric)
    True

    The metric matrix depends on position (r and theta):

    >>> p_sph = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 4, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> g_sph = metric_sph.metric_matrix(cx.charts.sph3d, p_sph)
    >>> # For physical (orthonormal) components, the metric is the identity
    >>> jnp.allclose(g_sph, jnp.eye(3))
    Array(True, dtype=bool)

    Get the metric for an embedded 2-sphere:

    >>> metric_sphere = cx.metrics.metric_of(cx.charts.twosphere)
    >>> isinstance(metric_sphere, cx.metrics.SphereMetric)
    True

    For Minkowski spacetime:

    >>> spacetime = cx.charts.SpaceTimeCT(cx.charts.cart3d)
    >>> cx.metrics.metric_of(spacetime)
    MinkowskiMetric()

    """
    raise NotImplementedError  # pragma: no cover


# ======================================================================
# Reference Frames


@plum.dispatch.abstract
def frame_of(obj: Any, /) -> Any:
    """Get the frame of an object."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def frame_transform_op(from_frame: Any, to_frame: Any, /) -> Any:
    """Make a frame transform.

    Parameters
    ----------
    from_frame : AbstractReferenceFrame
        The reference frame to transform from.
    to_frame : AbstractReferenceFrame
        The reference frame to transform to.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.ops as cxo
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> op
    Pipe((
      Translate(
          {'x': Q(i64[], 'km'), 'y': Q(i64[], 'km'), 'z': Q(i64[], 'km')},
          chart=Cart3D()
      ),
      Boost(
        {'x': Q(f64[], 'm / s'), 'y': Q(f64[], 'm / s'), 'z': Q(f64[], 'm / s')},
        Cart3D()
      )
    ))

    Apply to a {class}`coordinax.roles.PhysVel` vector at time tau=1 year:

    >>> v = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cx.roles.phys_vel)
    >>> t = u.Q(1, "yr")
    >>> result = op(t, v)
    >>> print(result)
    <Vector: chart=Cart3D, role=PhysVel (x, y, z) [km / s]
        [2.698e+05 2.000e+01 3.000e+01]>

    The Translate doesn't affect PhysVel (identity), and Boost adds its velocity offset.

    """
    raise NotImplementedError  # pragma: no cover


# ======================================================================
# Roles


# Defined here so that it can be re-exported in `coordinax.roles`
@plum.dispatch.abstract
def as_pos(x: Any, /) -> Any:
    r"""Convert a position vector to a displacement from some origin.

    Mathematical Definition
    -----------------------
    A **displacement** is defined relative to a reference point (origin).
    For position $p$ and origin $o$:

    $$
       \vec{d} = p - o \in T_o M
    $$
    The result is a tangent vector at the origin (or, in Euclidean space,
    a free vector).

    Parameters
    ----------
    x
        Position vector to convert. The full signature depends on the
        dispatched implementation.

    Returns
    -------
    displacement_vector
        A vector with ``PhsDisp`` role.

    See Also
    --------
    Vector.add : Add vectors with role semantics.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_role(obj: Any, /) -> "coordinax.roles.AbstractRole":
    """Infer role flag from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.roles as cxr

    >>> r1 = cxr.guess_role(u.dimension("length"))
    >>> r1
    Point()

    >>> r2 = cxr.guess_role(u.dimension("speed"))
    >>> r2
    PhysVel()

    >>> r3 = cxr.guess_role(u.dimension("acceleration"))
    >>> r3
    PhysAcc()

    """
    raise NotImplementedError  # pragma: no cover


# ==================================================================
# Transformations


@plum.dispatch.abstract
def point_transform(
    to_chart: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    from_chart: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    p: CsDict,
    /,
    usys: OptUSys = None,
) -> CsDict:
    r"""Transform position coordinates from one chart to another.

    This function implements coordinate transformations between different
    charts of the same geometric point in space. It is a point-wise
    map that preserves the physical location while changing the coordinate
    description.

    Mathematical Definition
    -----------------------
    Given position coordinates $q = (q^1, \ldots, q^n)$ in chart
    $\mathcal{R}_{\text{from}}$, compute the coordinates
    $p = (p^1, \ldots, p^m)$ in chart $\mathcal{R}_{\text{to}}$
    chartresenting the same physical point:

    $$
        p^i = f^i(q^1, \ldots, q^n), \quad i = 1, \ldots, m
    $$
    Common examples include:

    - **2D Polar → Cartesian**:

      $$
          x &= r \cos\theta \\
          y &= r \sin\theta
      $$
    - **3D Spherical → Cartesian**:

      $$
          x &= r \sin\theta \cos\phi \\
          y &= r \sin\theta \sin\phi \\
          z &= r \cos\theta
      $$
    - **3D Cylindrical → Cartesian**:

      $$
          x &= \rho \cos\phi \\
          y &= \rho \sin\phi \\
          z &= z
      $$

    Parameters
    ----------
    to_chart
        Target coordinate chart (e.g., ``cx.charts.cart3d``,
        ``cx.charts.sph3d``).  Defines the output coordinate system.
    from_chart
        Source coordinate chart (e.g., ``cx.charts.cyl3d``,
        ``cx.charts.polar2d``).  Defines the input coordinate system.
    p
        Dictionary of position coordinates in the source chart. Keys
        must match ``from_chart.components`` (e.g., ``"r"``, ``"theta"``,
        ``"phi"`` for spherical). Values must have appropriate physical
        dimensions.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    Mapping[str, Any]
        Dictionary of position coordinates in the target chart. Keys
        match ``to_chart.components`` and values preserve the physical
        dimensions appropriate for the target system.

    Raises
    ------
    NotImplementedError
        If no transformation rule is registered for the specific pair of
        charts ``(to_chart, from_chart)``.

    Notes
    -----
    - This is a **position-only** transformation. For velocities or
      accelerations, use ``physical_tangent_transform`` which accounts for the
      change of basis in the tangent space.

    - Transformations preserve physical dimensions. For example, converting from
      polar to Cartesian preserves that ``r`` has length dimension and produces
      ``x`` and ``y`` with length dimension.

    - Some transformations may introduce singularities (e.g., polar coordinates
      at the origin, spherical coordinates at poles). The transformation
      functions use ``arctan2`` and similar numerically stable functions where
      possible.

    - Transformations are composable: transforming $A \to B \to C$ yields the
      same result as a direct $A \to C$ transformation (up to numerical
      precision).

    - Identity transformations (same ``to_chart`` and ``from_chart``) return the
      input unchanged.

    See Also
    --------
    physical_tangent_transform : Transform velocity/acceleration in tangent space
    cartesian_chart : Get the Cartesian chart for a coordinate system

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Transform from 2D polar to Cartesian:

    >>> p_polar = {"r": u.Q(2.0, "m"), "theta": u.Angle(jnp.pi / 4, "rad")}
    >>> cxt.point_transform(cxc.cart2d, cxc.polar2d, p_polar)
    {'x': Quantity(Array(1.41421356, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(1.41421356, dtype=float64, ...), unit='m')}

    Transform from 3D spherical to Cartesian:

    >>> p_sph = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad"),
    ...          "r": u.Q(5.0, "km")}
    >>> cxt.point_transform(cxc.cart3d, cxc.sph3d, p_sph)
    {'x': Quantity(Array(5., dtype=float64, ...), unit='km'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='km'),
     'z': Quantity(Array(3.061617e-16, dtype=float64, ...), unit='km')}

    Transform from Cartesian to cylindrical:

    >>> p_xyz = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxt.point_transform(cxc.cyl3d, cxc.cart3d, p_xyz)
    {'rho': Quantity(Array(5., dtype=float64, ...), unit='m'),
     'phi': Quantity(Array(0.92729522, dtype=float64, ...), unit='rad'),
     'z': Quantity(Array(5., dtype=float64, ...), unit='m')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def physical_tangent_transform(
    to_chart: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    from_chart: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    v_phys: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    r"""Transform physical tangent components between coordinate charts.

    Overview:

    This transforms physical tangent-vector components (e.g., velocity or
    acceleration) from one coordinate chart to another, evaluated at a
    specific base point ``at``. Components are understood as physical components
    in an orthonormal frame with respect to the active metric for the
    chart, not as coordinate derivatives.

    Let $v$ be a tangent vector at point $q$ in chart $\mathcal R_\text{from}$
    and suppose the active metric for both charts is known via ``metric_of``.
    Define the orthonormal-frame matrices $B_\text{from}(q)$ and
    $B_\text{to}(p)$ whose columns are the orthonormal basis vectors expressed
    in the ambient or chart coordinates.  Then the physical components satisfy

    $$
        v_\text{cart} = B_\text{from}(q)\, v_\text{phys,from}
        \qquad\text{and}\qquad
        v_\text{phys,to} = B_\text{to}(p)^{\mathsf T} \, v_\text{cart},
    $$
    where $p$ is the same physical point expressed in
    $\mathcal R_\text{to}$, i.e.

    $$
        p = \text{point\_transform}(\mathcal R_\text{to}, \mathcal R_\text{from}, q).
    $$
    Eliminating $v_\text{cart}$ gives the core relation:

    $$
        v_\text{phys,to}
        = B_\text{to}(p)^{\mathsf T} \, B_\text{from}(q) \, v_\text{phys,from}.
    $$
    This is the operation implemented by ``physical_tangent_transform``.

    Important distinctions:

    - Physical vs coordinate derivatives: Inputs/outputs are uniform-dimension
        physical components (e.g., all speed or all acceleration). They are NOT
        coordinate time-derivatives, which may mix dimensions. For coordinate
        derivatives, see ``coord_transform``.
    - Position dependence: Frames depend on the evaluation point. The
        parameter ``at`` provides the base point in the ``from_chart`` chart.
    - Metrics: Orthonormal frames arise from the active metric resolved by
        ``metric_of``. Euclidean charts use Euclidean frames; spherical charts
        use the metric-induced frames, etc. Embedded manifolds use the ambient
        Euclidean metric for their tangent frames unless specified otherwise.

    Parameters
    ----------
    to_chart : coordinax.charts.AbstractChart
        Target coordinate chart whose physical components are desired.
    from_chart : coordinax.charts.AbstractChart
        Source coordinate chart in which ``v_phys`` is currently
        expressed as physical components.
    v_phys : CsDict
        Physical tangent components keyed by ``from_chart.components``. All
        values must share a consistent physical dimension (e.g., all speed for
        velocity, all acceleration for acceleration).
    at : CsDict
        Base-point coordinates where frames are evaluated, keyed by
        ``from_chart.components``. This point will be transformed to
        ``to_chart`` internally as needed.
    usys : unxt.AbstractUnitSystem, optional
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of
        light).

    Returns
    -------
    CsDict
        Physical tangent components keyed by ``to_chart.components`` with the
        same physical dimension as the input.

    Raises
    ------
    NotImplementedError
        If no transformation rule is registered for the specific pair of
        charts ``(to_chart, from_chart)``.
    ValueError
        If components in ``v_phys`` do not share a uniform physical dimension,
        or ``at`` does not provide a valid evaluation point.

    Notes
    -----
    - JAX compatibility: This is intended to operate on scalar-like leaves and
        composes with ``vmap``/``jit`` upstream. It does not assume batch shapes.
    - Embedded manifolds: ``embed_tangent`` and ``project_tangent`` are thin
        wrappers around this routine where one side is an ``EmbeddedManifold``
        and the other is the ambient chart.
    - Identity: If ``to_chart is from_chart``, the input is returned unchanged.

    See Also
    --------
    point_transform : Transform position coordinates between charts
    coord_transform : Transform coordinate time-derivatives
    embed_tangent : Embed intrinsic physical components into ambient components
    project_tangent : Project ambient physical components to intrinsic components
    metric_of : Resolve the active metric used to build orthonormal frames

    Examples
    --------
    Transform a velocity from Cartesian to spherical components (physical):

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    >>> q_cart = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
    >>> v_cart = {"x": u.Q(0.0, "km/s"), "y": u.Q(1.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> v_sph = cxt.physical_tangent_transform(
    ...     cxc.sph3d, cxc.cart3d, v_cart, at=q_cart
    ... )
    >>> v_sph
    {'phi': Quantity(Array(1., dtype=float64), unit='km / s'),
     'r': Quantity(Array(0., dtype=float64), unit='km / s'),
     'theta': Quantity(Array(0., dtype=float64), unit='km / s')}

    Transform an acceleration from spherical to Cartesian:

    >>> p_sph = {"theta": u.Q(jnp.pi/3, "rad"), "phi": u.Q(0.0, "rad"),
    ...          "r": u.Q(2.0, "km")}
    >>> a_sph = {"theta": u.Q(0.0, "km/s2"), "phi": u.Q(0.0, "km/s2"),
    ...          "r": u.Q(1.0, "km/s2")}
    >>> a_cart = cxt.physical_tangent_transform(
    ...     cxc.cart3d, cxc.sph3d, a_sph, at=p_sph)
    >>> a_cart
    {'x': Quantity(Array(0.8660254, dtype=float64), unit='km / s2'),
     'y': Quantity(Array(0., dtype=float64), unit='km / s2'),
     'z': Quantity(Array(0.5, dtype=float64), unit='km / s2')}

    Embedded manifold convenience via wrappers:

    >>> emb = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                            params={"R": u.Q(1.0, "km")})
    >>> p = {"theta": u.Angle(jnp.pi/2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}
    >>> v_cart2 = cxe.embed_tangent(emb, v_tan, at=p)
    >>> # uses physical_tangent_transform under the hood
    >>> cxe.project_tangent(emb, v_cart2, at=p)
    {'phi': Quantity(Array(0., dtype=float64), unit='km / s'),
     'theta': Quantity(Array(1., dtype=float64), unit='km / s')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def coord_transform(
    to_chart: "coordinax.chart.AbstractChart",
    from_chart: "coordinax.chart.AbstractChart",
    dqdt: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    r"""Transform coordinate-basis derivatives between charts.

    Maps coordinate time-derivatives (e.g., d𝜃/dt) between charts using the
    Jacobian of the point transformation.

    Mathematical Definition:
    For a coordinate transformation u = f(q), the coordinate derivatives transform via:

    $$
        \dot{u}^a = \frac{\partial f^a}{\partial q^i}(q) \dot{q}^i
    $$
    This is the standard chain rule for coordinate derivatives, unlike
    `physical_tangent_transform` which transforms physical (orthonormal frame)
    components.

    Parameters
    ----------
    to_chart : coordinax.charts.AbstractChart
        Target coordinate chart for the output components.
    from_chart : coordinax.charts.AbstractChart
        Source coordinate chart for the input components.
    dqdt : CsDict
        Dictionary of coordinate-derivative components (may have heterogeneous units).
        Keys match ``from_chart.components``.
    at : CsDict
        Position coordinates in the source chart. Keys match
        ``from_chart.components``.
    usys : unxt.AbstractUnitSystem, optional
        Unit system used when inputs are bare arrays (no units). When provided,
        coordinate dimensions are interpreted using ``from_chart.coord_dimensions``.

    Returns
    -------
    CsDict
        Dictionary of coordinate-derivative components in the target chart.
        Keys match ``to_chart.components``.

    Notes
    -----
    - **Heterogeneous units allowed**: Unlike `physical_tangent_transform`,
      components may have different physical dimensions (e.g., rad/s vs m/s).
    - **Jacobian-based**: Implementation uses the Jacobian of `point_transform`.
    - **Position-dependent**: Different points generally have different Jacobians.

    See Also
    --------
    physical_tangent_transform : Transform physical (orthonormal) components
    point_transform : Transform position coordinates

    """
    msg = (
        "No coord_transform rule registered for "
        f"{(type(to_chart), type(from_chart))!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch.abstract
def cotangent_transform(
    to_chart: "coordinax.chart.AbstractChart",
    from_chart: "coordinax.chart.AbstractChart",
    alpha: CsDict,
    /,
    *,
    at: CsDict,
) -> CsDict:
    r"""Transform cotangent (covector) components between charts.

    Maps 1-form / dual-vector components using the inverse Jacobian (pullback).

    Mathematical Definition:
    For a coordinate transformation u = f(q), cotangent components transform
    via:

    $$
        \\alpha'_a = \\alpha_i \\frac{\\partial q^i}{\\partial u^a}
    $$
    This is the pullback law for cotangent vectors (inverse of the Jacobian).

    Parameters
    ----------
    to_chart : coordinax.charts.AbstractChart
        Target coordinate chart for the output components.
    from_chart : coordinax.charts.AbstractChart
        Source coordinate chart for the input components.
    alpha : CsDict
        Dictionary of cotangent components. Keys match
        ``from_chart.components``.
    at : CsDict
        Position coordinates in the source chart. Keys match
        ``from_chart.components``.

    Returns
    -------
    CsDict
        Dictionary of cotangent components in the target chart.  Keys
        match ``to_chart.components``.

    Notes
    -----
    - **Pullback law**: Cotangents pull back (inverse Jacobian), unlike tangents
      which push forward (Jacobian).
    - **Metric-agnostic**: This transformation does not require the metric.  To
      convert between tangent and cotangent use `lower_index` / `raise_index`.

    See Also
    --------
    physical_tangent_transform : Transform physical tangent components
    lower_index : Convert tangent vectors to covectors via metric raise_index :
    Convert covectors to tangent vectors via metric

    """
    msg = (
        f"No cotangent_transform rule registered for "
        f"{(type(to_chart), type(from_chart))!r}."
    )
    raise NotImplementedError(msg)


@plum.dispatch.abstract
def physicalize(
    chart: "coordinax.chart.AbstractChart",
    dqdt: CsDict,
    /,
    *,
    at: CsDict,
) -> CsDict:
    r"""Convert coordinate derivatives to physical (orthonormal frame) components.

    Maps coordinate-basis derivatives (e.g., d𝜃/dt) to orthonormal frame
    components (e.g., velocity in m/s) using the scale factors of the coordinate
    system.

    Mathematical Definition:
    If E(q) maps coordinate-basis components to orthonormal physical components:

    $$
        v_{\\rm phys} = E(q) \\dot{q}
    $$

    Parameters
    ----------
    chart : coordinax.charts.AbstractChart
        Coordinate chart.
    dqdt : CsDict
        Coordinate-basis derivatives. Keys match ``chart.components``.
    at : CsDict
        Position coordinates. Keys match ``chart.components``.

    Returns
    -------
    CsDict
        Physical (orthonormal) components with homogeneous units.

    Notes
    -----
    - **Scale factors**: In orthogonal coordinates, E is diagonal with scale factors
      (e.g., h_r=1, h_θ=r, h_φ=r sin(θ) in spherical).
    - **Homogeneous output**: Unlike `coord_transform`, output has uniform units.

    See Also
    --------
    coordinateize : Inverse operation (physical to coordinate derivatives)
    physical_tangent_transform : Transform between orthonormal frames

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def coordinateize(
    chart: "coordinax.chart.AbstractChart",
    v_phys: CsDict,
    /,
    *,
    at: CsDict,
) -> CsDict:
    r"""Convert physical (orthonormal frame) components to coordinate derivatives.

    Maps orthonormal frame components (e.g., velocity in m/s) to coordinate-basis
    derivatives (e.g., d𝜃/dt) using the inverse of the scale-factor matrix.

    Mathematical Definition:
    If E(q) maps coordinate-basis to orthonormal components:

    $$
        \\dot{q} = E(q)^{-1} v_{\\rm phys}
    $$

    Parameters
    ----------
    chart : coordinax.charts.AbstractChart
        Coordinate chart.
    v_phys : CsDict
        Physical (orthonormal) components. Keys match ``chart.components``.
    at : CsDict
        Position coordinates. Keys match ``chart.components``.

    Returns
    -------
    CsDict
        Coordinate-basis derivatives (may have heterogeneous units).

    Notes
    -----
    - **Scale factors**: Inverse of those in `physicalize`.
    - **Heterogeneous output**: Output components may have different units.

    See Also
    --------
    physicalize : Inverse operation (coordinate to physical components)
    coord_transform : Transform coordinate derivatives between charts

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def lower_index(
    chart: "coordinax.chart.AbstractChart",
    v_coord: CsDict,
    /,
    *,
    at: CsDict,
) -> CsDict:
    r"""Convert coordinate-basis tangent vectors to cotangent (1-form) components.

    Uses the metric tensor to lower indices: $\alpha_i = g_{ij} v^j$.

    Mathematical Definition:
    $$
        \\alpha_i = g_{ij} v^j
    $$
    where g is the metric tensor of the chart.

    Parameters
    ----------
    chart : coordinax.charts.AbstractChart
        Coordinate chart.
    v_coord : CsDict
        Coordinate-basis tangent components. Keys match ``chart.components``.
    at : CsDict
        Position coordinates for metric evaluation. Keys match ``chart.components``.

    Returns
    -------
    CsDict
        Cotangent (1-form) components. Keys match ``chart.components``.

    See Also
    --------
    raise_index : Inverse operation (covector to vector)
    metric_of : Get the metric tensor for a chart

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def raise_index(
    chart: "coordinax.chart.AbstractChart",
    alpha: CsDict,
    /,
    *,
    at: CsDict,
) -> CsDict:
    r"""Convert cotangent (1-form) components to coordinate-basis tangent vectors.

    Uses the inverse metric tensor to raise indices: $v^i = g^{ij} \alpha_j$.

    Mathematical Definition
    -----------------------
    $$
        v^i = g^{ij} \\alpha_j
    $$
    where g^{-1} is the inverse metric tensor.

    Parameters
    ----------
    chart : coordinax.charts.AbstractChart
        Coordinate chart.
    alpha : CsDict
        Cotangent (1-form) components. Keys match ``chart.components``.
    at : CsDict
        Position coordinates for metric evaluation. Keys match ``chart.components``.

    Returns
    -------
    CsDict
        Coordinate-basis tangent components. Keys match ``chart.components``.

    See Also
    --------
    lower_index : Inverse operation (vector to covector)
    metric_of : Get the metric tensor for a chart

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def frame_cart(
    chart: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> Array:
    r"""Return an orthonormal frame expressed in ambient Cartesian components.

    Mathematical definition:
    $$
       B(q) = [\hat e_1(q)\ \cdots\ \hat e_n(q)]
       \\
       \hat e_i \cdot \hat e_j = \delta_{ij} \quad \text{(Euclidean)}
    $$

    Parameters
    ----------
    chart
        Chart whose orthonormal frame is requested.
    at
        Coordinate values keyed by ``chart.components``.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    Array
        Matrix of shape ``(n_{\text{ambient}}, n_{\text{chart}})`` with columns
        equal to the orthonormal frame vectors expressed in ambient Cartesian
        components.

    Notes
    -----
    - For Euclidean 3D charts, ``n_ambient = n_chart = 3``.
    - For embedded manifolds, the frame is rectangular (e.g. ``3\times 2`` for ``S^2``).
    - For ``SpaceTimeCT``, orthonormality is with respect to the Minkowski metric
      with signature ``(-,+,+,+)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> p = {
    ...     "r": u.Q(1.0, "km"),
    ...     "theta": u.Angle(1.0, "rad"),
    ...     "phi": u.Angle(0.5, "rad"),
    ... }
    >>> B = cx.transforms.frame_cart(cx.charts.sph3d, at=p)
    >>> bool(jnp.allclose(B.T @ B, jnp.eye(3)))
    True

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pushforward(frame_basis: Any, v_chart: Any, /) -> Any:
    """Push forward components from a chart frame into Cartesian components."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pullback(
    metric: "coordinax.metrics.AbstractMetric", frame_basis: Any, v_cart: Any, /
) -> Any:
    """Pull back Cartesian components into chart-frame components."""
    raise NotImplementedError  # pragma: no cover


# ==================================================================
# Operators


@plum.dispatch.abstract
def apply_op(op: "coordinax.ops.AbstractOperator", tau: Any, x: Any, /) -> Any:
    r"""Apply an operator to an input at a given time.

    This is the core dispatch function for operator application. Each operator
    type registers its own implementation via multiple dispatch. Operators act
    on various input types (Quantity, Vector, CsDict) according to their
    semantics.

    Mathematical Definition
    -----------------------
    For an operator $\mathcal{O}$ parameterized by time $\tau$, this computes:

    $$
        x' = \mathcal{O}(\tau)(x)
    $$

    For time-independent operators, $\tau$ is ignored. For composite operators
    (e.g., ``Pipe``, ``GalileanOp``), the component operators are applied
    sequentially.

    Parameters
    ----------
    op : coordinax.ops.AbstractOperator
        The operator to apply. This can be any operator type:

        - ``Translate``: Spatial translation (Point role)
        - ``Boost``: Velocity offset (Vel role)
        - ``Rotate``: Spatial rotation
        - ``Identity``: No-op
        - ``Pipe``: Sequential composition
        - ``GalileanOp``: Full Galilean transformation

    tau : Any
        Time parameter for time-dependent operators. Pass ``None`` for
        time-independent operators. For time-dependent operators, this is
        used to evaluate callable parameters (e.g., ``Translate(lambda t: ...)``)
        via ``eval_op``.

    x : Any
        The input to transform. Supported types depend on the operator:

        - ``Quantity``: Direct arithmetic application
        - ``Vector``: Role-aware transformation with chart preservation
        - ``CsDict``: Low-level component dict (requires ``role=`` kwarg)

    **kwargs : Any
        Additional keyword arguments passed to the dispatch:

        - ``role``: Required for CsDict inputs to specify geometric role
        - ``at``: Base point for non-Euclidean transformations (future)

    Returns
    -------
    Any
        The transformed input, same type as ``x``. For role-specialized
        operators (``Translate``, ``Boost``), the role of the output matches the
        input.

    Raises
    ------
    TypeError
        If a role-specialized operator is applied to an incompatible role.
        For example, applying ``Translate`` to a ``PhysVel``-role vector raises
        ``TypeError``.
    NotImplementedError
        If no dispatch is registered for the given ``(operator, input)`` types.

    Notes
    -----
    - **Role enforcement**: ``Translate`` only acts on ``Point`` role,
      ``Boost`` only on ``PhysVel`` role.
      This ensures geometric correctness (points translate, velocities boost).

    - **Operator.__call__**: The ``__call__`` method of operators delegates
      to this function: ``op(tau, x)`` is equivalent to ``apply_op(op, tau, x)``.

    - **Time evaluation**: For operators with callable parameters, ``eval_op``
      is called internally to materialize the time-dependent values.

    - **Composite operators**: For ``Pipe`` and ``GalileanOp``, the component
      operators are applied in sequence (left-to-right for Pipe).

    See Also
    --------
    coordinax.ops.eval_op : Evaluate time-dependent operator parameters
    coordinax.ops.simplify : Simplify operators to canonical form
    coordinax.ops.Translate : Point translation operator
    coordinax.ops.Boost : Velocity boost operator

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    **Apply a translation to a Quantity:**

    >>> shift = cxo.Translate.from_([1, 2, 3], "km")
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxo.apply_op(shift, None, q)
    Quantity(Array([1, 2, 3], dtype=int64), unit='km')

    **Apply a boost to a velocity Quantity:**

    >>> boost = cxo.Boost.from_([100, 0, 0], "km/s")
    >>> v = u.Q([0, 50, 0], "km/s")
    >>> cxo.apply_op(boost, None, v)
    Quantity(Array([100,  50,   0], dtype=int64), unit='km / s')

    **Apply a rotation:**

    >>> import jax.numpy as jnp
    >>> Rz = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> rot = cxo.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "m")
    >>> cxo.apply_op(rot, None, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='m')

    **Composite operator (GalileanOp):**

    >>> op = cxo.GalileanOp(
    ...     translation=cxo.Translate.from_([1, 0, 0], "km"),
    ...     velocity=cxo.Boost.from_([0, 0, 0], "km/s"),
    ... )
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxo.apply_op(op, None, q)
    Quantity(Array([1., 0., 0.], dtype=float64), unit='km')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def simplify(op: Any, /) -> Any:
    """Simplify an operator to a canonical form.

    This function takes an operator and attempts to simplify it, returning a
    new, potentially simpler operator. For example, a ``Translate`` with zero
    delta simplifies to ``Identity``.

    Parameters
    ----------
    op : AbstractOperator
        The operator to simplify.

    Returns
    -------
    AbstractOperator
        A simplified operator. May be a different type (e.g., ``Identity``)
        if the original operator has no effect.

    Notes
    -----
    This function uses multiple dispatch. Each operator type registers its
    own simplification rules.

    To see all available dispatches::

        >>> import coordinax.ops as cxo
        >>> cxo.simplify.methods  # doctest: +ELLIPSIS
        List of ... method(s):
        ...

    Examples
    --------
    >>> import coordinax.ops as cxo

    **Identity (already simple):**

    >>> op = cxo.Identity()
    >>> cxo.simplify(op) is op
    True

    **Translate with zero delta:**

    >>> op = cxo.Translate.from_([0, 0, 0], "m")
    >>> cxo.simplify(op)
    Identity()

    **Translate with non-zero delta (no simplification):**

    >>> op = cxo.Translate.from_([1, 2, 3], "m")
    >>> simplified = cxo.simplify(op)
    >>> type(simplified).__name__
    'Translate'

    **Rotate with identity matrix:**

    >>> import unxt as u
    >>> op = cxo.Rotate.from_euler("z", u.Q(0, "deg"))
    >>> cxo.simplify(op)
    Identity()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def cdict(obj: Any, /) -> CsDict:
    """Extract component dictionary from an object.

    This function converts various coordinate charts into a component
    dictionary where keys are component names and values are the corresponding
    values.

    Parameters
    ----------
    obj
        An object to extract a component dictionary from. Supported types include:
        - Vector: extracted from ``obj.data``
        - unxt.Quantity: treated as Cartesian coordinates with components in the
          last dimension, matched to the appropriate Cartesian chart
        - Mappings: returned as-is

    Returns
    -------
    dict[str, Any]
        A mapping from component names to values.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Extract from a Vector:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    Extract from a Quantity treated as Cartesian:

    >>> q = u.Q([1, 2, 3], "m")
    >>> d = cx.cdict(q)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    raise NotImplementedError  # pragma: no cover
