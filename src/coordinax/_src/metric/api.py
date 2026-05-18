"""Concrete fallback dispatch rules for metric_matrix and metric_representation.

The dispatch function objects are defined (as abstract) in
:mod:`coordinax.api.manifolds`.  This module registers the concrete fallback
rules that apply to any :class:`~coordinax._src.base.AbstractManifold` /
:class:`~coordinax._src.base.AbstractChart` pair not covered by a more specific
rule registered in a ``register_metric.py`` module.

Importing this module is sufficient to ensure both fallback rules are active;
the ``register_metric.py`` modules import from here so that these rules are
always present before any specific rule is added.
"""

__all__ = ("metric_matrix", "metric_representation")

import plum

from .matrix import AbstractMetricMatrix, DenseMetric
from coordinax._src.base import AbstractChart, AbstractManifold


@plum.dispatch
def metric_matrix(
    M: AbstractManifold, point: dict, chart: AbstractChart, /
) -> AbstractMetricMatrix:
    """Fallback — raise with a helpful message for unregistered pairs.

    Concrete ``(manifold, chart)`` pairs register their own dispatch rules in
    the relevant ``register_metric.py`` modules (loaded as part of Phase 2).

    Parameters
    ----------
    M : AbstractManifold
        The manifold carrying the metric field.
    point : CDict
        A component dictionary giving the coordinates in ``chart``.
    chart : AbstractChart
        The coordinate chart in which to express the metric.

    Raises
    ------
    NotImplementedError
        When no specific dispatch rule is registered for the given types.

    """
    del point
    msg = (
        f"No metric_matrix dispatch registered for "
        f"manifold={type(M).__name__!r}, chart={type(chart).__name__!r}. "
        f"Register a rule with @plum.dispatch on metric_matrix."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def metric_representation(
    M: AbstractManifold, chart: AbstractChart, /
) -> type[AbstractMetricMatrix]:
    """Return `DenseMetric` as the default fallback.

    More specific rules (e.g. for Cartesian charts) override this and return
    `DiagonalMetric`.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> from coordinax.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DenseMetric
    >>> from coordinax._src.charts.d3 import LonCosLatSpherical3D

    Non-orthogonal chart falls back to ``DenseMetric``:

    >>> metric_representation(cxm.R3, LonCosLatSpherical3D()) is DenseMetric
    True

    """
    del M, chart
    return DenseMetric
