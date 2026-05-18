"""`coordinax.manifolds` module."""

__all__ = (
    # API functions
    "guess_manifold",
    "pt_embed",
    "pt_project",
    "pt_map",
    "scale_factors",
    "angle_between",
    "metric_matrix",
    "metric_representation",
    # Abstract Manifold/Atlas/Metric
    "AbstractAtlas",
    "AbstractMetricField",
    "AbstractManifold",
    "AbstractDiagonalMetricField",
    # Metric matrix classes
    "AbstractMetricMatrix",
    "DiagonalMetric",
    "DenseMetric",
    # Null
    "NoManifold",
    "no_manifold",
    "NoMetric",
    "no_metric",
    "NoAtlas",
    "no_atlas",
    # Euclidean
    "EuclideanAtlas",
    "FlatMetric",
    "EuclideanManifold",
    "Rn",
    "R0",
    "R1",
    "R2",
    "R3",
    "RN",
    # HyperSpherical
    "HyperSphericalAtlas",
    "RoundMetric",
    "HyperSphericalManifold",
    "Sn",
    "S1",
    "S2",
    # Minkowski
    "MinkowskiAtlas",
    "MinkowskiMetric",
    "MinkowskiManifold",
    "minkowski4d",
    # Product
    "CartesianProductAtlas",
    "ProductMetric",
    "CartesianProductManifold",
    # Embeddings
    "EmbeddedManifold",
    "AbstractEmbeddingMap",
    "CustomEmbeddingMap",
    "TwoSphereIn3D",
    "embedded_twosphere",
    "EmbeddedManifold",
    "EmbeddedChart",
    "PullbackMetric",
    # Custom
    "CustomAtlas",
    "CustomMetric",
    "CustomManifold",
    # Product / Galilean
    "galilean_spacetime",
)

from ._src.setup_package import install_import_hook

with install_import_hook("coordinax.manifolds"):
    from ._src.base import (
        AbstractAtlas,
        AbstractDiagonalMetricField,
        AbstractManifold,
        AbstractMetricField,
    )
    from ._src.custom import CustomAtlas, CustomManifold, CustomMetric
    from ._src.embedded import (
        AbstractEmbeddingMap,
        CustomEmbeddingMap,
        EmbeddedChart,
        EmbeddedManifold,
        PullbackMetric,
    )
    from ._src.euclidean import (
        R0,
        R1,
        R2,
        R3,
        RN,
        EuclideanAtlas,
        EuclideanManifold,
        FlatMetric,
        Rn,
    )
    from ._src.manifolds import *  # noqa: F403
    from ._src.metric import AbstractMetricMatrix, DenseMetric, DiagonalMetric
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
        ProductMetric,
    )
    from ._src.product.galilean_ct import galilean_spacetime
    from ._src.spherical import (
        S1,
        S2,
        HyperSphericalAtlas,
        HyperSphericalManifold,
        RoundMetric,
        Sn,
        TwoSphereIn3D,
        embedded_twosphere,
    )
    from coordinax.api.charts import pt_map
    from coordinax.api.manifolds import (
        angle_between,
        guess_manifold,
        metric_matrix,
        metric_representation,
        pt_embed,
        pt_project,
        scale_factors,
    )


del install_import_hook
