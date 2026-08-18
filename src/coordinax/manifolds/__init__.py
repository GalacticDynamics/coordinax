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
    "norm",
    "chord_distance",
    "geodesic_distance",
    "interval",
    # Sub-namespaces
    "lorentzian",
    # Abstract Manifold/Atlas/Metric
    "AbstractAtlas",
    "AbstractMetricField",
    "AbstractManifold",
    "AbstractDiagonalMetricField",
    "AbstractLorentzianMetricField",
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

from coordinax._src.setup_package import install_import_hook

with install_import_hook("coordinax.manifolds"):
    from . import lorentzian
    from coordinax._src.base import (
        AbstractAtlas,
        AbstractDiagonalMetricField,
        AbstractLorentzianMetricField,
        AbstractManifold,
        AbstractMetricField,
    )
    from coordinax._src.custom import CustomAtlas, CustomManifold, CustomMetric
    from coordinax._src.embedded import (
        AbstractEmbeddingMap,
        CustomEmbeddingMap,
        EmbeddedChart,
        EmbeddedManifold,
        PullbackMetric,
    )
    from coordinax._src.euclidean import (
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
    from coordinax._src.manifolds import *
    from coordinax._src.metric import AbstractMetricMatrix, DenseMetric, DiagonalMetric
    from coordinax._src.minkowski import (
        MinkowskiAtlas,
        MinkowskiManifold,
        MinkowskiMetric,
        minkowski4d,
    )
    from coordinax._src.null import (
        NoAtlas,
        NoManifold,
        NoMetric,
        no_atlas,
        no_manifold,
        no_metric,
    )
    from coordinax._src.product import (
        CartesianProductAtlas,
        CartesianProductManifold,
        ProductMetric,
    )
    from coordinax._src.product.galilean_ct import galilean_spacetime
    from coordinax._src.spherical import (
        S1,
        S2,
        HyperSphericalAtlas,
        HyperSphericalManifold,
        RoundMetric,
        Sn,
        TwoSphereIn3D,
        embedded_twosphere,
    )
    from coordinaxs.api.charts import pt_map
    from coordinaxs.api.manifolds import (
        angle_between,
        chord_distance,
        geodesic_distance,
        guess_manifold,
        interval,
        metric_matrix,
        metric_representation,
        norm,
        pt_embed,
        pt_project,
        scale_factors,
    )


del install_import_hook
