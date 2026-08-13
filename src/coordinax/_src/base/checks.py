"""Preconditions shared by the metric-level dispatches."""

__all__ = ("check_metric_is_charts",)

from .charts import AbstractChart
from .metric import AbstractMetricField


def check_metric_is_charts(
    metric: AbstractMetricField, chart: AbstractChart, fname: str, /
) -> None:
    """Refuse a metric that is not the one ``chart`` carries.

    The metric gates the dispatch; the primitives underneath all read
    ``chart.M.metric``, so a differing one cannot be honoured and must not be
    quietly replaced by the chart's.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> v = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    >>> try:
    ...     cxm.norm(v, cxm.MinkowskiMetric(), cxc.cart3d, at=at)
    ... except ValueError as e:
    ...     print(str(e)[:52])
    norm(): metric-level dispatch needs the chart's own

    """
    if metric != chart.M.metric:
        msg = (
            f"{fname}(): metric-level dispatch needs the chart's own metric; "
            f"got {metric} for a chart carrying {chart.M.metric}. The metric "
            f"selects the method; it is not applied in place of the chart's."
        )
        raise ValueError(msg)
