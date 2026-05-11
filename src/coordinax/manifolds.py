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
    "NoManifold",
    "no_manifold",
    # Abstract Manifold/Atlas/Metric
    "AbstractAtlas",
    "AbstractMetric",
    "AbstractManifold",
    "AbstractDiagonalMetric",
    # Null
    "NoAtlas",
    "no_atlas",
    # Euclidean
    "EuclideanAtlas",
    "EuclideanMetric",
    "EuclideanManifold",
    "Rn",
    "euclidean3d",
    # HyperSpherical
    "HyperSphericalAtlas",
    "HyperSphericalMetric",
    "HyperSphericalManifold",
    "Sn",
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
    from ._src.base_atlas import AbstractAtlas, NoAtlas, no_atlas
    from ._src.base_manifold import AbstractManifold
    from ._src.base_metric import AbstractDiagonalMetric, AbstractMetric
    from ._src.custom import CustomAtlas, CustomManifold, CustomMetric
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
        Rn,
        euclidean3d,
    )
    from ._src.manifolds import *  # noqa: F403
    from ._src.minkowski import (
        MinkowskiAtlas,
        MinkowskiManifold,
        MinkowskiMetric,
        minkowski4d,
    )
    from ._src.no_manifold import NoManifold, no_manifold
    from ._src.product import (
        CartesianProductAtlas,
        CartesianProductManifold,
        CartesianProductMetric,
    )
    from ._src.spherical import (
        HyperSphericalAtlas,
        HyperSphericalManifold,
        HyperSphericalMetric,
        Sn,
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
