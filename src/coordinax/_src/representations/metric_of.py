"""Metric resolver for representations."""

__all__ = ("metric_of",)

import plum

from .metrics import AbstractMetric, EuclideanMetric, MinkowskiMetric, SphereMetric
from coordinax._src.representations import euclidean as r
from coordinax._src.representations.embed import EmbeddedManifold
from coordinax._src.representations.manifolds import TwoSphere
from coordinax._src.representations.spacetime import SpaceTimeCT


@plum.dispatch
def metric_of(rep: r.AbstractDimensionalFlag, /) -> AbstractMetric:
    r"""Resolve the default metric for a representation.

    Mathematical definition
    -----------------------
    .. math::
       \mathrm{metric\_of}(\mathrm{rep}) = g_{\mathrm{rep}}
       \\
       g_{\mathrm{rep}} \text{ is the default metric for the chart}

    Parameters
    ----------
    rep
        Representation (rep) whose default metric is requested.

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
    >>> g = cx.r.metric_of(cx.r.twosphere).metric_matrix(cx.r.twosphere, p)
    >>> g.shape
    (2, 2)

    """
    return EuclideanMetric(rep.dimensionality)


@plum.dispatch
def metric_of(rep: TwoSphere, /) -> AbstractMetric:
    """Return the intrinsic metric for the two-sphere chart."""
    return SphereMetric()


@plum.dispatch
def metric_of(rep: SpaceTimeCT, /) -> AbstractMetric:
    """Return the Minkowski metric for spacetime."""
    return MinkowskiMetric()


@plum.dispatch
def metric_of(rep: r.SpaceTimeEuclidean, /) -> AbstractMetric:
    """Return the Euclidean metric for 4D spacetime."""
    return EuclideanMetric(4)


@plum.dispatch
def metric_of(rep: EmbeddedManifold, /) -> AbstractMetric:
    """Return the ambient Euclidean metric for embedded manifolds."""
    return metric_of(rep.ambient_kind.cartesian)
