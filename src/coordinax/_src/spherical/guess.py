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
    """Return the manifold of a two-sphere chart *class*.

    `guess_chart` dispatches on the class, not an instance, so without this the
    two-sphere charts fell through to the `type[AbstractChart]` fallback and
    were built carrying `NoManifold()`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.SphericalTwoSphere)
    HyperSphericalManifold(ndim=2)

    So a mapping of angles now infers the sphere rather than the sentinel:

    >>> cxm.guess_manifold({"theta": 1.0, "phi": 0.5})
    HyperSphericalManifold(ndim=2)

    """
    return HyperSphericalManifold(2)


@plum.dispatch
def guess_manifold(_: type[CircularOneSphere], /) -> HyperSphericalManifold:
    """Return the manifold of the circle chart *class*.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.guess_manifold(cxc.CircularOneSphere)
    HyperSphericalManifold(ndim=1)

    """
    return HyperSphericalManifold(1)
