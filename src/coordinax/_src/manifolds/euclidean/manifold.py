"""Euclidean manifolds."""

__all__ = ("EuclideanManifold", "euclidean3d")

from dataclasses import dataclass

from typing import final

import jax

from .atlas import EuclideanAtlas
from .metric import EuclideanMetric
from coordinax._src.manifolds.base import AbstractManifold


@jax.tree_util.register_static
@final
@dataclass(frozen=True, slots=True)
class EuclideanManifold(AbstractManifold):
    r"""The $n$-dimensional Euclidean manifold $\mathbb{R}^n$.

    The **Euclidean manifold** of dimension $n$ is the smooth manifold
    $(\mathbb{R}^n, \mathcal{A}_{\mathbb{R}^n})$.

    **Charts and atlas.** The smooth structure is described by an
    {class}`coordinax.manifolds.EuclideanAtlas` whose charts are local
    diffeomorphisms

    $$\varphi : U \subset \mathbb{R}^n \to \mathbb{R}^n.$$

    A chart $C = (U, \varphi)$ is admitted by the atlas when its dimensionality
    matches $n$ and either (1) it is explicitly registered with
    {class}`coordinax.manifolds.EuclideanAtlas`, or (2) it possesses a
    compatible transition map to the default Cartesian chart. For $n = 3$ the
    built-in charts include Cartesian $(x, y, z)$, spherical $(r, \theta,
    \phi)$, cylindrical $(\rho, \phi, z)$, and several angular variants; see
    {class}`coordinax.manifolds.EuclideanAtlas` for the full list.

    **Transition maps.** For any two charts $C_\alpha = (U_\alpha,
    \varphi_\alpha)$ and $C_\beta = (U_\beta, \varphi_\beta)$ in the atlas, the
    **transition map** is

    $$\tau_{\alpha \to \beta}
        = \varphi_\beta \circ \varphi_\alpha^{-1} : \varphi_\alpha(U_\alpha \cap
        U_\beta)
          \to \varphi_\beta(U_\alpha \cap U_\beta).$$

    Because $\mathbb{R}^n$ is flat and simply connected, every transition map is
    a smooth diffeomorphism on a connected open domain. For example, the
    Cartesian-to-spherical transition on $\mathbb{R}^3$ is

    $$\tau_{C \to S}(x, y, z)
        = \Bigl(\sqrt{x^2 + y^2 + z^2},\;
                \arccos\!\tfrac{z}{r},\; \operatorname{atan2}(y, x)\Bigr).$$

    **Pre-built instance.** The module exports
    {obj}`coordinax.manifolds.euclidean3d` as a pre-built instance for the
    common case $\mathbb{R}^3$.

    Parameters
    ----------
    ndim : int
        Intrinsic dimension $n \geq 0$ of the manifold — the number of
        independent coordinates required to label a point.

    Attributes
    ----------
    atlas : EuclideanAtlas
        The atlas of coordinate charts compatible with this manifold. Its
        {attr}`~EuclideanAtlas.ndim` equals ``ndim``.

    Examples
    --------
    **Construction**

    Construct a Euclidean manifold of arbitrary dimension:

    >>> import coordinax.manifolds as cxmd
    >>> M = cxmd.EuclideanManifold(3)
    >>> M
    EuclideanManifold(ndim=3)

    The intrinsic dimension is accessible via {attr}`~EuclideanManifold.ndim`:

    >>> M.ndim
    3

    The atlas is a {class}`EuclideanAtlas` with matching dimensionality:

    >>> M.atlas
    EuclideanAtlas(ndim=3)

    **Default chart**

    The default chart is the standard Cartesian chart for the given dimension:

    >>> M.default_chart()
    Cart3D()

    >>> cxmd.EuclideanManifold(2).default_chart()
    Cart2D()

    >>> cxmd.EuclideanManifold(1).default_chart()
    Cart1D()

    **Chart membership**

    Check whether a chart belongs to this manifold's atlas:

    >>> import coordinax.charts as cxc

    >>> M.has_chart(cxc.cart3d)
    True

    >>> M.has_chart(cxc.sph3d)
    True

    Charts with the wrong dimensionality are rejected:

    >>> M.has_chart(cxc.cart2d)
    False

    {meth}`check_chart` raises if the chart is not supported:

    >>> try:
    ...     M.check_chart(cxc.cart2d)
    ... except ValueError as e:
    ...     print(e)
    Chart Cart2D() is not supported by this manifold atlas.

    **Point transition maps**

    Convert a point from Cartesian to spherical coordinates on $\mathbb{R}^3$.
    The point $(0, 0, 1)$ maps to $(r, \theta, \phi) = (1, 0, 0)$:

    >>> x = {"x": 0.0, "y": 0.0, "z": 1.0}
    >>> M.pt_map(x, cxc.cart3d, cxc.sph3d)
    {'r': Array(1., ...), 'theta': Array(0., ...), 'phi': Array(0., ...)}

    Transitioning a point from Cartesian to polar coordinates on $\mathbb{R}^2$.
    The point $(1, 1)$ has distance $\sqrt{2}$ and angle $\pi/4$ from the
    origin:

    >>> M2 = cxmd.EuclideanManifold(2)
    >>> x2 = {"x": 1.0, "y": 1.0}
    >>> M2.pt_map(x2, cxc.cart2d, cxc.polar2d)
    {'r': Array(1.41421356, ...), 'theta': Array(0.78539816, ...)}

    **Pre-built instances**

    For the most common case — three-dimensional Euclidean space $\mathbb{R}^3$
    — the module provides a pre-built instance:

    >>> cxmd.euclidean3d
    EuclideanManifold(ndim=3)

    """

    ndim: int
    """Intrinsic dimension of the manifold."""

    def __init__(self, ndim: int, /) -> None:
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "atlas", EuclideanAtlas(self.ndim))
        object.__setattr__(self, "metric", EuclideanMetric(self.ndim))


euclidean3d = EuclideanManifold(3)
r"""The 3-dimensional Euclidean manifold, i.e. $\mathbb{R}^3$."""
