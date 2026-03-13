"""`coordinax.manifolds` module."""

__all__ = (
    # Manifolds
    "AbstractManifold",
    "EuclideanManifold",
    "TwoSphereManifold",
    # Atlases
    "AbstractAtlas",
    "EuclideanAtlas",
    "TwoSphereAtlas",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.manifolds"):
    from ._src import (
        AbstractAtlas,
        AbstractManifold,
        EuclideanAtlas,
        EuclideanManifold,
        TwoSphereAtlas,
        TwoSphereManifold,
    )


del install_import_hook
