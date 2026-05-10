"""`coordinax.manifolds` module."""

__all__ = (
    # API functions
    "guess_manifold",
    "pt_embed",
    "pt_project",
    "pt_map",
    "scale_factors",
    "angle_between",
    # Topological Space
    "AbstractTopologicalManifold",
    "NoManifold",
    "no_manifold",
    # Abstract Manifold/Atlas/Metric
    "AbstractAtlas",
    "AbstractMetric",
    "AbstractManifold",
    "AbstractDiagonalMetric",
    # Euclidean
    "EuclideanAtlas",
    "EuclideanMetric",
    "EuclideanManifold",
    "euclidean3d",
    # HyperSpherical
    "HyperSphericalAtlas",
    "HyperSphericalMetric",
    "HyperSphericalManifold",
    "twosphere",
    # Minkowski
    "MinkowskiAtlas",
    "MinkowskiMetric",
    "MinkowskiManifold",
    "minkowski4d",
    # Product
    "CartesianProductAtlas",
    "CartesianProductMetric",
    "CartesianProductManifold",
    # Embeddings
    "EmbeddedManifold",
    "AbstractEmbeddingMap",
    "CustomEmbeddingMap",
    "TwoSphereIn3D",
    "embedded_twosphere",
    "EmbeddedManifold",
    "EmbeddedChart",
    "InducedMetric",
    # Custom
    "CustomAtlas",
    "CustomMetric",
    "CustomManifold",
)

from ._src.setup_package import install_import_hook

with install_import_hook("coordinax.manifolds"):
    from ._src.base_atlas import AbstractAtlas
    from ._src.base_manifold import AbstractManifold
    from ._src.base_metric import AbstractDiagonalMetric, AbstractMetric
    from ._src.base_topo import AbstractTopologicalManifold, NoManifold, no_manifold
    from ._src.embedded import (
        AbstractEmbeddingMap,
        CustomEmbeddingMap,
        EmbeddedChart,
        EmbeddedManifold,
        InducedMetric,
    )
    from ._src.euclidean import (
        EuclideanAtlas,
        EuclideanManifold,
        EuclideanMetric,
        euclidean3d,
    )
    from ._src.manifolds import CustomAtlas, CustomManifold, CustomMetric
    from ._src.minkowski import (
        MinkowskiAtlas,
        MinkowskiManifold,
        MinkowskiMetric,
        minkowski4d,
    )
    from ._src.product import (
        CartesianProductAtlas,
        CartesianProductManifold,
        CartesianProductMetric,
    )
    from ._src.spherical import (
        HyperSphericalAtlas,
        HyperSphericalManifold,
        HyperSphericalMetric,
        TwoSphereIn3D,
        embedded_twosphere,
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
