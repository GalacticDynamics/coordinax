"""Manifolds in coordinax."""

__all__: tuple[str, ...] = ()


import plum

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
