"""Metric resolver for representations."""

__all__ = ("metric_of",)

import plum

import coordinax._src.charts as cxc
import coordinax._src.embed as cxe
from .metrics import AbstractMetric, EuclideanMetric, MinkowskiMetric, SphereMetric
from coordinax._src import api


@plum.dispatch
def metric_of(chart: cxc.AbstractDimensionalFlag, /) -> AbstractMetric:
    r"""Resolve the default metric for a representation.

    Mathematical definition:
    $$
       \mathrm{metric\_of}(\mathrm{rep}) = g_{\mathrm{rep}}
       \\
       g_{\mathrm{rep}} \text{ is the default metric for the chart}
    $$

    Parameters
    ----------
    chart
        Chart whose default metric is requested.

    Returns
    -------
    AbstractMetric
        Metric instance appropriate for ``rep``.

    Notes
    -----
    - Euclidean reps default to ``EuclideanMetric``.
    - ``TwoSphere`` defaults to ``SphereMetric``.
    - ``SpaceTimeCT`` defaults to ``MinkowskiMetric`` with signature ``(-,+,+,+)``.
    - ``SpaceTimeEuclidean`` defaults to 4D Euclidean metric.
    - ``EmbeddedManifold`` uses the ambient Euclidean metric for physical components.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> p = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> metric = cx.metrics.metric_of(cx.charts.twosphere)
    >>> g = metric.metric_matrix(cx.charts.twosphere, p)
    >>> g.shape
    (2, 2)

    """
    return EuclideanMetric(chart.ndim)


@plum.dispatch
def metric_of(chart: cxc.TwoSphere, /) -> AbstractMetric:
    """Return the intrinsic metric for the two-sphere chart."""
    return SphereMetric()


@plum.dispatch
def metric_of(chart: cxc.SpaceTimeCT, /) -> AbstractMetric:  # type: ignore[type-arg]
    """Return the Minkowski metric for spacetime."""
    # TODO: spatial chart isn't always 3d
    return MinkowskiMetric()


@plum.dispatch
def metric_of(chart: cxc.SpaceTimeEuclidean, /) -> AbstractMetric:  # type: ignore[type-arg]
    """Return the Euclidean metric for 4D spacetime."""
    return EuclideanMetric(n=len(chart.components))


@plum.dispatch
def metric_of(chart: cxe.EmbeddedManifold, /) -> AbstractMetric:  # type: ignore[type-arg]
    """Return the ambient Euclidean metric for embedded manifolds."""
    return api.metric_of(chart.ambient_chart.cartesian)
