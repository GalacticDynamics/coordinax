"""Vector API for coordinax."""

__all__ = (
    "cartesian_chart",
    "point_transition_map",
    # Realization maps
    "point_realization_map",
    "realize_cartesian",
    "unrealize_cartesian",
    # Data
    "cdict",
    "guess_chart",
)

from typing import TYPE_CHECKING, Any

import plum

from ._custom_types import CDict

if TYPE_CHECKING:
    import coordinax.charts  # noqa: ICN001


@plum.dispatch.abstract
def cartesian_chart(obj: Any, /) -> "coordinax.charts.AbstractChart":  # type: ignore[type-arg]
    r"""Return the corresponding Cartesian chart for a given chart.

    This function provides the canonical flat-space Cartesian chart
    associated with any coordinate system. It maps each chart to its
    natural Cartesian equivalent in the same dimensional space.

    Mathematical Definition:

    For a coordinate chart $\mathcal{R}$ in $n$-dimensional
    space, this returns the Cartesian chart $\mathcal{C}_n$ such that:

    $$ \mathrm{cartesian\_chart}(\mathcal{R}) = \mathcal{C}_n $$

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
      returns the chart type. Use {func}`~coordinax.charts.point_realization_map` for actual
      coordinate conversion.
    - All standard Euclidean coordinate systems (spherical, cylindrical, polar)
      map to their dimensional Cartesian equivalent.
    - For embedded manifolds, this returns the Cartesian form of the ambient
      space, not the intrinsic coordinates.

    See Also
    --------
    point_realization_map : Transform coordinates between charts
    coordinax.charts.AbstractChart : Base class for coordinate charts

    Examples
    --------
    >>> import coordinax.charts as cxc

    1D coordinate systems map to Cart1D:

    >>> cxc.cartesian_chart(cxc.cart1d)
    Cart1D(...)

    >>> cxc.cartesian_chart(cxc.radial1d)
    Cart1D(...)

    2D coordinate systems map to Cart2D:

    >>> cxc.cartesian_chart(cxc.cart2d)
    Cart2D(...)

    >>> cxc.cartesian_chart(cxc.polar2d)
    Cart2D(...)

    3D coordinate systems map to Cart3D:

    >>> cxc.cartesian_chart(cxc.cart3d)
    Cart3D(...)

    >>> cxc.cartesian_chart(cxc.sph3d)
    Cart3D(...)

    >>> cxc.cartesian_chart(cxc.cyl3d)
    Cart3D(...)

    N-dimensional systems map to CartND:

    >>> cxc.cartesian_chart(cxc.cartnd)
    CartND(...)

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def point_transition_map(*args: Any, **kwargs: Any) -> CDict:
    r"""Transform point coordinates between charts on the same manifold.

    This function implements the ordinary **chart transition map** between
    two charts belonging to the same atlas of the same manifold. It is the
    intrinsic coordinate-change operation: the underlying point on the
    manifold is unchanged, and only its coordinate representation is changed.

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

    Notes
    -----
    - This is a **point-only** map.
    - This function is restricted to charts on the **same manifold** and,
      conceptually, the same intrinsic geometry.
    - In particular, it does **not** cover more general coordinate mappings
      that pass between different manifolds or between intrinsic and ambient
      realizations of an embedded manifold.
    - For those more general point-coordinate mappings, use `point_realization_map`.

    See Also
    --------
    point_realization_map : More general point-coordinate map, including cross-manifold
        realization-style mappings.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def point_realization_map(*args: Any, **kwargs: Any) -> CDict:
    r"""Transform position coordinates from one chart to another.

    This function implements the most general point-coordinate map between
    two compatible chart representations of the same geometric point. It is a
    point-wise map that preserves the physical location while changing the
    coordinate description.

    Unlike `point_transition_map`, this function is not restricted to two charts on
    the same manifold. It may also represent a **realization-style** map
    between charts attached to different manifolds when one is a realization
    of the other, such as an intrinsic chart on an embedded manifold and a
    chart on its ambient manifold. In that case, this function may change
    both the chart and the manifold in which the point is being represented.

    Mathematical Definition:

    Given position coordinates $q = (q^1, \ldots, q^n)$ in chart
    $\mathcal{R}_{\text{from}}$, compute the coordinates
    $p = (p^1, \ldots, p^m)$ in chart $\mathcal{R}_{\text{to}}$
    representing the same geometric point:

    $$
        p^i = f^i(q^1, \ldots, q^n), \quad i = 1, \ldots, m
    $$

    When ``to_chart`` and ``from_chart`` are charts on the same manifold,
    `point_realization_map` reduces to the ordinary chart transition map handled by
    `point_transition_map`.

    More generally, if $\varphi_{\mathrm{from}} : U \subset M \to \mathbb{R}^n$
    and $\psi_{\mathrm{to}} : W \subset N \to \mathbb{R}^m$ are chart maps on
    manifolds $M$ and $N$, and there is a point map
    $F : M \supset U \to W \subset N$, then `point_realization_map` represents the
    coordinate expression

    $$
        \psi_{\mathrm{to}} \circ F \circ \varphi_{\mathrm{from}}^{-1}.
    $$

    Raises
    ------
    NotImplementedError
        If no transformation rule is registered for the specific pair of
        charts ``(to_chart, from_chart)``.

    Notes
    -----
    - This is a **position-only** transformation.

    - This function may map between charts on the same manifold or across
      manifolds, provided a compatible point map is defined between them.
    - Same-manifold chart changes are the special case handled by
      `point_transition_map`.
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
    point_transition_map : transform position coordinates between charts on the same manifold.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Transform from 2D polar to Cartesian:

    >>> p_polar = {"r": u.Q(2.0, "m"), "theta": u.Angle(jnp.pi / 4, "rad")}
    >>> cxc.point_realization_map(cxc.cart2d, cxc.polar2d, p_polar)
    {'x': Q(1.41421356, 'm'), 'y': Q(1.41421356, 'm')}

    Transform from 3D spherical to Cartesian:

    >>> p_sph = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad"),
    ...          "r": u.Q(5.0, "km")}
    >>> cxc.point_realization_map(cxc.cart3d, cxc.sph3d, p_sph)
    {'x': Q(5., 'km'), 'y': Q(0., 'km'), 'z': Q(3.061617e-16, 'km')}

    Transform from Cartesian to cylindrical:

    >>> p_xyz = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxc.point_realization_map(cxc.cyl3d, cxc.cart3d, p_xyz)
    {'rho': Q(5., 'm'), 'phi': Q(0.92729522, 'rad'), 'z': Q(5., 'm')}

    """
    del args, kwargs  # Unused in abstract method
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def realize_cartesian(*args: Any, **kwargs: Any) -> Any:
    r"""Realize a point in canonical ambient Cartesian coordinates.

    This method evaluates the chart's (optional) **ambient realization map**

    $$ X: V \subset \mathbb{R}^n \to \mathbb{R}^m $$

    mapping point-role coordinates in this chart to point-role coordinates
    in the chart's distinguished ambient Cartesian chart,
    ``self.cartesian``.

    - This is **point-role only**. It does not apply to tangent-valued roles
      (e.g. ``CoordVel``/``PhysVel``).
    - For parameter-free Euclidean reparameterizations (e.g. spherical or
      cylindrical charts on $\mathbb{R}^3$), this is typically a canonical
      map.
    - For charts whose realization depends on additional geometric data
      (e.g. a 2-sphere in $\mathbb{R}^3$ requiring a radius), charts may not
      provide a canonical realization; in such cases the underlying
      transition rule to ``self.cartesian`` may be unregistered and this
      method will fail. In that case, users should use
      `coordinax.manifolds.AbstractManifold.realize_cartesian` with an
      appropriate embedded manifold chart or construct a
      `coordinax.embeddings.EmbeddedChart` which provides a custom
      realization map.

    Raises
    ------
    Exception
        If no point transition rule exists from this chart to
        ``self.cartesian`` (e.g. for charts requiring embedding parameters).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Spherical to ambient Cartesian:

    >>> at = {"r": u.Q(2.0, "m"), "theta": u.Q(1.5707963, "rad"),
    ...       "phi": u.Q(0.0, "rad")}
    >>> cxc.sph3d.realize_cartesian(at)
    {'x': Q(2., 'm'), 'y': Q(0., 'm'), 'z': Q(5...e-08, 'm')}

    Cartesian to Cartesian is the identity:

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"),
    ...       "z": u.Q(3.0, "m")}
    >>> cxc.cart3d.realize_cartesian(at)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    del args, kwargs  # Unused in abstract method
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def unrealize_cartesian(*args: Any, **kwargs: Any) -> Any:
    """Invert the ambient Cartesian realization on the chart domain.

    This method applies the inverse of the chart's ambient realization map
    (when defined) to convert point-role coordinates in ``self.cartesian``
    to point-role coordinates in this chart.

    - This is **point-role only**.
    - The inverse map may be undefined or multi-valued globally; this method
        represents the inverse only on the chart's intended domain.
    - For charts whose realization depends on additional geometric data
        (e.g. embedding parameters), the corresponding transition rule from
        ``self.cartesian`` may be unregistered and this method will fail. In
        that case, users should use
        `coordinax.manifolds.AbstractManifold.realize_cartesian` with an
        appropriate embedded manifold chart or construct a
        `coordinax.embeddings.EmbeddedChart` which provides a custom
        realization map.

    Raises
    ------
    Exception
        If no point transition rule exists from ``self.cartesian`` to this
        chart.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Cartesian point back to spherical:

    >>> cart_pt = {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"),
    ...           "z": u.Q(0.0, "m")}
    >>> cxc.sph3d.unrealize_cartesian(cart_pt)
    {'r': Q(2., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(1.57079633, 'rad')}

    Cartesian to Cartesian is the identity:

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"),
    ...       "z": u.Q(3.0, "m")}
    >>> cxc.cart3d.unrealize_cartesian(at)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    del args, kwargs  # Unused in abstract method
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_chart(*_: Any) -> "coordinax.charts.AbstractChart":  # type: ignore[type-arg]
    """Infer a Cartesian chart from the shape of a value / quantity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> cxc.guess_chart(frozenset(("x", "y", "z")))
    Cart3D()

    >>> cxc.guess_chart({"x": 1.0, "y": 2.0, "z": 3.0})
    Cart3D()

    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cxc.guess_chart(q)
    Cart3D()

    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> cxc.guess_chart(x)
    Cart3D()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def cdict(_: Any, /) -> CDict:
    """Extract component dictionary from an object.

    This function converts various coordinate charts into a component dictionary
    where keys are component names and values are the corresponding values.

    Parameters
    ----------
    obj
        An object to extract a component dictionary from.
        Dispatch rules include:

        - `collections.abc.Mapping`: returned as-is
        - `unxt.Quantity`: treated as Cartesian coordinates with components in
          the last dimension, matched to the appropriate Cartesian chart

    Returns
    -------
    dict[str, Any]
        A mapping from component names to values.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import unxt as u

    Extract from a Mapping:

    >>> d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> cx.cdict(d)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cx.cdict(d)
    {'x': 1.0, 'y': 2.0, 'z': 3.0}

    Extract from a `unxt.Quantity` treated as Cartesian:

    >>> q = u.Q([1, 2, 3], "m")
    >>> cx.cdict(q)
    {'x': Q(1, 'm'), 'y': Q(2, 'm'), 'z': Q(3, 'm')}

    Specify the chart for a `unxt.Quantity`. For homogeneous unit Quantities,
    this must be Cartesian:

    >>> cx.cdict(cx.cart3d, q)
    {'x': Q(1, 'm'), 'y': Q(2, 'm'), 'z': Q(3, 'm')}

    Extract from an Array-like object with a registered chart:

    >>> arr = jnp.array([1.0, 2.0, 3.0])
    >>> cx.cdict(cx.cart3d, arr)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    raise NotImplementedError  # pragma: no cover
