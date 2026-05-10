"""Manifolds in coordinax."""

__all__: tuple[str, ...] = ()


import plum

import coordinax.charts as cxc
from .atlas import EuclideanAtlas
from .manifold import EuclideanManifold


@plum.dispatch
def guess_manifold(obj: EuclideanAtlas, /) -> EuclideanManifold:
    """Return the manifold of a Euclidean atlas.

    >>> import coordinax.manifolds as cxm
    >>> atlas = cxm.EuclideanAtlas(3)
    >>> cxm.guess_manifold(atlas)
    Rn(3)

    """
    return EuclideanManifold(obj.ndim)


@plum.dispatch
def guess_manifold(obj: cxc.Cart0D, /) -> EuclideanManifold:
    """Return a EuclideanManifold for 0-D charts.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.Cart0D())
    Rn(0)

    """
    return EuclideanManifold(0)


@plum.dispatch
def guess_manifold(obj: cxc.Cart1D | cxc.Radial1D, /) -> EuclideanManifold:
    """Return a EuclideanManifold for 1-D charts.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.Cart1D())
    Rn(1)

    """
    return EuclideanManifold(1)


@plum.dispatch
def guess_manifold(obj: cxc.Cart2D | cxc.Polar2D, /) -> EuclideanManifold:
    """Return a EuclideanManifold for 2-D charts.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.Cart2D())
    Rn(2)

    """
    return EuclideanManifold(2)


@plum.dispatch
def guess_manifold(
    obj: cxc.Cart3D
    | cxc.Cylindrical3D
    | cxc.AbstractSpherical3D
    | cxc.ProlateSpheroidal3D,
    /,
) -> EuclideanManifold:
    """Return a EuclideanManifold for 3-D charts.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.Cart3D())
    Rn(3)

    """
    return EuclideanManifold(3)
