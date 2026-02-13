"""`coordinax.metrics` Module."""

__all__ = (
    "AbstractMetric",
    "EuclideanMetric",
    "MinkowskiMetric",
    "SphereMetric",
    "metric_of",
    "norm",
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.metrics"):
    from ._src import (
        AbstractMetric,
        EuclideanMetric,
        MinkowskiMetric,
        SphereMetric,
        norm,
    )
    from coordinax.api import metric_of


del setup_package
