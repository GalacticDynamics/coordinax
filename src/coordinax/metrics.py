"""`coordinax.metrics` Module."""

__all__ = (
    "AbstractMetric",
    "EuclideanMetric",
    "MinkowskiMetric",
    "SphereMetric",
    "metric_of",
    "norm",
    "raise_index",
    "lower_index",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.metrics"):
    from ._src.api import lower_index, metric_of, raise_index
    from ._src.metrics import (
        AbstractMetric,
        EuclideanMetric,
        MinkowskiMetric,
        SphereMetric,
        norm,
    )


del install_import_hook, RUNTIME_TYPECHECKER
