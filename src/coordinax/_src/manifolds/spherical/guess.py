"""Manifolds in coordinax."""

__all__: tuple[str, ...] = ()


import plum

import coordinax.charts as cxc
from .atlas import HyperSphericalAtlas
from .manifold import HyperSphericalManifold


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
def guess_manifold(obj: cxc.AbstractSphericalTwoSphere, /) -> HyperSphericalManifold:
    """Return a HyperSphericalManifold manifold.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.SphericalTwoSphere())
    HyperSphericalManifold(ndim=2)

    """
    return HyperSphericalManifold(obj.ndim)
