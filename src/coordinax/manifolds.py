"""`coordinax.manifolds` module."""

__all__ = (
    # API functions
    "guess_manifold",
    "pt_embed",
    "pt_project",
    "pt_map",
    "scale_factors",
    "angle_between",
    # Abstract Manifold/Atlas/Metric
    "AbstractAtlas",
    "AbstractMetric",
    "AbstractManifold",
    "AbstractDiagonalMetric",
    # Null
    "NoManifold",
    "no_manifold",
    "NoMetric",
    "no_metric",
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
    "onesphere",
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
    from ._src.base import (
        AbstractAtlas,
        AbstractDiagonalMetric,
        AbstractManifold,
        AbstractMetric,
    )
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
    from ._src.null import (
        NoAtlas,
        NoManifold,
        NoMetric,
        no_atlas,
        no_manifold,
        no_metric,
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
        Sn,
        TwoSphereIn3D,
        embedded_twosphere,
        onesphere,
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
