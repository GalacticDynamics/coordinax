"""Vector API for coordinax."""

__all__ = ("embed_point", "project_point", "embed_tangent", "project_tangent")

import plum

import unxt as u

from ._custom_types import CsDict


@plum.dispatch.abstract
def embed_point(
    embedded: object, p_pos: CsDict, /, *, usys: u.AbstractUnitSystem | None = None
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

    >>> embedding_chart = cxe.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
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
def project_point(*args: object, usys: u.AbstractUnitSystem | None = None) -> CsDict:
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

    >>> sphere = cxe.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
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
    >>> cxe.project_point(sphere, p_south)
    {'theta': Quantity(Array(3.14159265, dtype=float64, ...), unit='rad'),
    'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def embed_tangent(
    embedded: object,
    v_chart: CsDict,
    /,
    *,
    at: CsDict,
    usys: u.AbstractUnitSystem | None = None,
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

    >>> sphere = cxe.EmbeddedManifold(
    ...     intrinsic_chart=cxc.twosphere,
    ...     ambient_chart=cxc.cart3d,
    ...     params={"R": u.Q(1.0, "km")},
    ... )

    Define a point and a tangent vector (velocity) at that point:

    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_chart = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}

    Embed the tangent vector to ambient Cartesian components:

    >>> v_ambient = cxe.embed_tangent(sphere, v_chart, at=p)
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
    embedded: object,
    v_ambient: CsDict,
    /,
    *,
    at: CsDict,
    usys: u.AbstractUnitSystem | None = None,
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

    >>> sphere = cxe.EmbeddedManifold(
    ...     intrinsic_chart=cxc.twosphere,
    ...     ambient_chart=cxc.cart3d,
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

    >>> v_chart = cxe.project_tangent(sphere, v_ambient, at=p)
    >>> jnp.allclose(v_chart["theta"].value, -0.5, atol=1e-10)
    Array(True, dtype=bool)
    >>> jnp.allclose(v_chart["phi"].value, 1.0, atol=1e-10)
    Array(True, dtype=bool)

    Note that the normal component (radial $z$) is discarded.

    Verify round-trip property:

    >>> v_chart_orig = {"theta": u.Q(2.0, "km/s"), "phi": u.Q(3.0, "km/s")}
    >>> v_amb = cxe.embed_tangent(sphere, v_chart_orig, at=p)
    >>> v_recovered = cxe.project_tangent(sphere, v_amb, at=p)
    >>> jnp.allclose(
    ...     u.uconvert("km/s", v_recovered["theta"]).value,
    ...     u.uconvert("km/s", v_chart_orig["theta"]).value,
    ... )
    Array(True, dtype=bool)

    """
    raise NotImplementedError  # pragma: no cover
