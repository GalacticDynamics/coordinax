"""`coordinax.manifolds` module."""

__all__ = (
    # API functions
    "guess_manifold",
    "pt_embed",
    "pt_project",
    "pt_map",
    # Manifolds
    "AbstractManifold",
    "EmbeddedManifold",
    "EuclideanManifold",
    "euclidean3d",
    "HyperSphericalManifold",
    "twosphere",
    "CartesianProductManifold",
    "CustomManifold",
    "MinkowskiManifold",
    # Atlases
    "AbstractAtlas",
    "EuclideanAtlas",
    "HyperSphericalAtlas",
    "CartesianProductAtlas",
    "CustomAtlas",
    "MinkowskiAtlas",
    # Embeddings
    "AbstractEmbeddingMap",
    "CustomEmbeddingMap",
    "TwoSphereIn3D",
    "embedded_twosphere",
    # Convenience chart
    "EmbeddedChart",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.manifolds"):
    from ._src import (
        AbstractAtlas,
        AbstractEmbeddingMap,
        AbstractManifold,
        CartesianProductAtlas,
        CartesianProductManifold,
        CustomAtlas,
        CustomEmbeddingMap,
        CustomManifold,
        EmbeddedChart,
        EmbeddedManifold,
        EuclideanAtlas,
        EuclideanManifold,
        HyperSphericalAtlas,
        HyperSphericalManifold,
        MinkowskiAtlas,
        MinkowskiManifold,
        TwoSphereIn3D,
        embedded_twosphere,
        euclidean3d,
        twosphere,
    )
    from coordinax.api.charts import pt_map
    from coordinax.api.manifolds import guess_manifold, pt_embed, pt_project


del install_import_hook
