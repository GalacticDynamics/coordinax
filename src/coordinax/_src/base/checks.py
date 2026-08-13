"""Preconditions shared by the metric-level dispatches."""

__all__ = ("check_metric_is_charts",)

from .charts import AbstractChart
from .metric import AbstractMetricField


def check_metric_is_charts(
    metric: AbstractMetricField, chart: AbstractChart, fname: str, /
) -> None:
    """Refuse a metric that is not the one ``chart`` carries.

    The primitives underneath all read ``chart.M.metric``, so a differing one
    cannot be honoured and must not be quietly swapped for the chart's.

    A manifold that leaves its dimension open -- ``Rn(N)``, spelled
    ``ndim is True`` -- pins nothing, so its metric is unbound and equality is
    the wrong test. There the caller's metric supplies the concrete dimension
    and only the *kind* can be required: `cxc.cartnd` accepts any
    `FlatMetric`, and still refuses a `MinkowskiMetric`.

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
    chart_metric = chart.M.metric
    ok = (
        isinstance(metric, type(chart_metric))
        if getattr(chart.M, "ndim", None) is True
        else metric == chart_metric
    )
    if not ok:
        msg = (
            f"{fname}(): metric-level dispatch needs the chart's own metric; "
            f"got {metric} for a chart carrying {chart_metric}. The metric "
            f"selects the method; it is not applied in place of the chart's."
        )
        raise ValueError(msg)
