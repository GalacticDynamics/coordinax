"""Vector API for coordinax."""

__all__ = (
    "physicalize",
    "coordinateize",
    "point_transform",
    "physical_tangent_transform",
    "coord_transform",
    "frame_cart",
    "pushforward",
    "pullback",
)

from jaxtyping import Array

import plum

import unxt as u

from ._custom_types import CsDict


@plum.dispatch.abstract
def point_transform(
    to_chart: object,
    from_chart: object,
    p: CsDict,
    /,
    usys: u.AbstractUnitSystem | None = None,
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
        Target coordinate chart (e.g., ``cxc.cart3d``,
        ``cxc.sph3d``).  Defines the output coordinate system.
    from_chart
        Source coordinate chart (e.g., ``cxc.cyl3d``,
        ``cxc.polar2d``).  Defines the input coordinate system.
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
    usys: u.AbstractUnitSystem | None = None,
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

    >>> emb = cxe.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
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
    usys: u.AbstractUnitSystem | None = None,
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
def frame_cart(
    chart: "coordinax.charts.AbstractChart",  # type: ignore[type-arg]
    /,
    *,
    at: CsDict,
    usys: u.AbstractUnitSystem | None = None,
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
    >>> B = cxt.frame_cart(cxc.sph3d, at=p)
    >>> bool(jnp.allclose(B.T @ B, jnp.eye(3)))
    True

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pushforward(frame_basis: object, v_chart: object, /) -> object:
    """Push forward components from a chart frame into Cartesian components."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pullback(metric: object, frame_basis: object, v_cart: object, /) -> object:
    """Pull back Cartesian components into chart-frame components."""
    raise NotImplementedError  # pragma: no cover
