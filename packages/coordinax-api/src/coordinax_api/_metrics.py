"""Vector API for coordinax."""

__all__ = ("metric_of",)


from typing import TYPE_CHECKING, Any

import plum

if TYPE_CHECKING:
    import coordinax.metrics  # noqa: ICN001


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
        ``cxc.sph3d``, ``cxc.cart3d``), but the function supports
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
    >>> import coordinax.charts as cxc
    >>> import coordinax.metrics as cxm
    >>> import unxt as u

    Get the metric for 3D Cartesian coordinates (Euclidean):

    >>> metric_cart = cxm.metric_of(cxc.cart3d)
    >>> isinstance(metric_cart, cxm.EuclideanMetric)
    True

    The metric matrix is the identity:

    >>> g_cart = metric_cart.metric_matrix(cxc.cart3d, {})
    >>> g_cart.shape
    (3, 3)
    >>> import jax.numpy as jnp
    >>> jnp.allclose(g_cart, jnp.eye(3))
    Array(True, dtype=bool)

    Get the metric for 3D spherical coordinates:

    >>> metric_sph = cxm.metric_of(cxc.sph3d)
    >>> isinstance(metric_sph, cxm.EuclideanMetric)
    True

    The metric matrix depends on position (r and theta):

    >>> p_sph = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 4, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> g_sph = metric_sph.metric_matrix(cxc.sph3d, p_sph)
    >>> # For physical (orthonormal) components, the metric is the identity
    >>> jnp.allclose(g_sph, jnp.eye(3))
    Array(True, dtype=bool)

    Get the metric for an embedded 2-sphere:

    >>> metric_sphere = cxm.metric_of(cxc.twosphere)
    >>> isinstance(metric_sphere, cxm.SphereMetric)
    True

    For Minkowski spacetime:

    >>> spacetime = cxc.SpaceTimeCT(cxc.cart3d)
    >>> cxm.metric_of(spacetime)
    MinkowskiMetric()

    """
    raise NotImplementedError  # pragma: no cover
