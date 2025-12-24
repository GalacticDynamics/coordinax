"""`coordinax.metrics` Module."""

__all__ = (
    "AbstractMetric",
    "EuclideanMetric",
    "MinkowskiMetric",
    "SphereMetric",
    "metric_of",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.metrics"):
    from ._src.metrics import (
        AbstractMetric,
        EuclideanMetric,
        MinkowskiMetric,
        SphereMetric,
        metric_of,
    )


del install_import_hook, RUNTIME_TYPECHECKER
