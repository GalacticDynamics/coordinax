"""Euclidean manifolds."""

__all__ = (
    "EuclideanManifold",
    "Rn",
    "R0",
    "R1",
    "R2",
    "R3",
    "RN",
)

import dataclasses

from typing import Any, Final, final

import jax
import wadler_lindig as wl

import dataclassish

from .atlas import EuclideanAtlas
from .metric import EuclideanMetric
from coordinax._src.base import AbstractManifold
from coordinax._src.internal import pos_named_objs


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
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
    {obj}`coordinax.manifolds.R3` as a pre-built instance for the
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
    Rn(3)

    The intrinsic dimension is accessible via {attr}`~EuclideanManifold.ndim`:

    >>> M.ndim
    3

    The atlas is a {class}`EuclideanAtlas` with matching dimensionality:

    >>> M.atlas
    EuclideanAtlas(ndim=3)

    **Default chart**

    The default chart is the standard Cartesian chart for the given dimension:

    >>> M.default_chart()
    Cart3D(M=Rn(3))

    >>> cxmd.EuclideanManifold(2).default_chart()
    Cart2D(M=Rn(2))

    >>> cxmd.EuclideanManifold(1).default_chart()
    Cart1D(M=Rn(1))

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
    Chart Cart2D(M=Rn(2)) is not supported by this manifold atlas.

    **Pre-built instances**

    For the most common case — three-dimensional Euclidean space $\mathbb{R}^3$
    — the module provides a pre-built instance:

    >>> cxmd.R3
    Rn(3)

    """

    ndim: int
    """Intrinsic dimension of the manifold."""

    def __init__(self, ndim: int, /) -> None:
        # Check `ndim` is a positive integer or True. True works for Rn(N),
        # deferring the check until the default chart is requested.
        if ndim is not True and not (
            isinstance(ndim, int) and not isinstance(ndim, bool) and ndim >= 0
        ):
            msg = f"`ndim` must be True or a non-negative integer, got {ndim!r}"
            raise TypeError(msg)
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "atlas", EuclideanAtlas(self.ndim))
        object.__setattr__(self, "metric", EuclideanMetric(self.ndim))

    def __pdoc__(self, *, alias: bool = True, **kw: Any) -> wl.AbstractDoc:
        """Return the string representation.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(3)
        >>> wl.pprint(M)
        Rn(3)

        >>> wl.pformat(M, alias=False)
        'EuclideanManifold(3)'

        """
        name = "Rn" if alias else "EuclideanManifold"
        docs = pos_named_objs(
            list(dataclassish.field_items(self)),
            ("ndim",),
            self.__dataclass_fields__,
            **kw,
        )
        return wl.bracketed(
            begin=wl.TextDoc(f"{name}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )


Rn = EuclideanManifold
"""Alias for `EuclideanManifold`."""


R0: Final = EuclideanManifold(0)
r"""The 0-dim Euclidean manifold, i.e. $\mathbb{R}^0$."""

R1: Final = EuclideanManifold(1)
r"""The 1-dim Euclidean manifold, i.e. $\mathbb{R}^1$."""

R2: Final = EuclideanManifold(2)
r"""The 2-dim Euclidean manifold, i.e. $\mathbb{R}^2$."""

R3: Final = EuclideanManifold(3)
r"""The 3-dim Euclidean manifold, i.e. $\mathbb{R}^3$."""

RN: Final = EuclideanManifold(True)  # noqa: FBT003
r"""The $n$-dim Euclidean manifold, i.e. $\mathbb{R}^n$ for any $n \geq 0$."""
