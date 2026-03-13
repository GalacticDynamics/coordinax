"""`coordinax.metrics` Module."""

__all__ = (
    "AbstractMetric",
    "EuclideanMetric",
    "MinkowskiMetric",
    "SphereMetric",
    "norm",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.metrics"):
    from ._src import (
        AbstractMetric,
        EuclideanMetric,
        MinkowskiMetric,
        SphereMetric,
        norm,
    )


del install_import_hook
