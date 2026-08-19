"""Manifolds in coordinax."""

__all__: tuple[str, ...] = ()


import plum

from .atlas import HyperSphericalAtlas
from .chart import AbstractSphericalTwoSphere, CircularOneSphere, SphericalTwoSphere
from .manifold import HyperSphericalManifold
from coordinax._src.charts.register_guess import register_canonical_chart

# `("theta", "phi")` is also `MathSphericalTwoSphere`, which swaps the polar and
# azimuthal angles; component names cannot tell them apart, so name the one
# `guess_chart` should infer. Declared here rather than in
# `coordinax._src.charts`, which is imported before this package and so cannot
# reach `SphericalTwoSphere`.
register_canonical_chart(SphericalTwoSphere)


@plum.dispatch
def guess_manifold(obj: HyperSphericalAtlas, /) -> HyperSphericalManifold:
    """Return the manifold of a HyperSphericalAtlas.

    >>> import coordinax.manifolds as cxm
    >>> atlas = cxm.HyperSphericalAtlas()
    >>> cxm.guess_manifold(atlas)
    HyperSphericalManifold(ndim=2)

    """
    return HyperSphericalManifold(obj.ndim)


@plum.dispatch
def guess_manifold(obj: AbstractSphericalTwoSphere, /) -> HyperSphericalManifold:
    """Return a HyperSphericalManifold manifold.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.SphericalTwoSphere())
    HyperSphericalManifold(ndim=2)

    """
    return HyperSphericalManifold(obj.ndim)


@plum.dispatch
def guess_manifold(_: type[AbstractSphericalTwoSphere], /) -> HyperSphericalManifold:
    """Return a HyperSphericalManifold manifold, from the chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.SphericalTwoSphere)
    HyperSphericalManifold(ndim=2)

    """
    return HyperSphericalManifold(2)


@plum.dispatch
def guess_manifold(_: type[CircularOneSphere], /) -> HyperSphericalManifold:
    """Return a HyperSphericalManifold manifold, from the chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.CircularOneSphere)
    HyperSphericalManifold(ndim=1)

    """
    return HyperSphericalManifold(1)
