"""`coordinax.manifolds` module."""

__all__ = (
    # API functions
    "guess_manifold",
    "pt_embed",
    "pt_project",
    "pt_map",
    "scale_factors",
    "angle_between",
    # Metrics
    "AbstractMetric",
    "CartesianProductMetric",
    "CustomMetric",
    "EuclideanMetric",
    "HyperSphericalMetric",
    "InducedMetric",
    "MinkowskiMetric",
    # Manifolds
    "AbstractManifold",
    "EmbeddedManifold",
    "EuclideanManifold",
    "euclidean3d",
    "HyperSphericalManifold",
    "twosphere",
    "MinkowskiManifold",
    "minkowski4d",
    "CartesianProductManifold",
    "CustomManifold",
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
    "EmbeddedManifold",
    # Convenience chart
    "EmbeddedChart",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.manifolds"):
    from ._src import (
        AbstractAtlas,
        AbstractEmbeddingMap,
        AbstractManifold,
        AbstractMetric,
        CartesianProductAtlas,
        CartesianProductManifold,
        CartesianProductMetric,
        CustomAtlas,
        CustomEmbeddingMap,
        CustomManifold,
        CustomMetric,
        EmbeddedChart,
        EmbeddedManifold,
        EuclideanAtlas,
        EuclideanManifold,
        EuclideanMetric,
        HyperSphericalAtlas,
        HyperSphericalManifold,
        HyperSphericalMetric,
        InducedMetric,
        MinkowskiAtlas,
        MinkowskiManifold,
        MinkowskiMetric,
        TwoSphereIn3D,
        embedded_twosphere,
        euclidean3d,
        minkowski4d,
        twosphere,
    )
    from coordinax.api.charts import pt_map
    from coordinax.api.manifolds import (
        angle_between,
        guess_manifold,
        pt_embed,
        pt_project,
        scale_factors,
    )


del install_import_hook
