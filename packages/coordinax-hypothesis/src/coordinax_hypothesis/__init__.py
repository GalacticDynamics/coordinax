"""Hypothesis strategies for coordinax."""

__all__ = (
    "angles",
    "pointedvectors",
    "distances",
    "distance_moduli",
    "parallaxes",
    "can_point_transform",
    "chart_classes",
    "chart_init_kwargs",
    "build_init_kwargs_strategy",
    "charts",
    "charts_like",
    "chart_time_chain",
    "product_charts",
    "cdicts",
    "physical_roles",
    "point_role",
    "roles",
    "coord_roles",
    "vectors",
    "vectors_with_target_chart",
)

from ._src.angles import angles
from ._src.cdict import cdicts
from ._src.charts import (
    build_init_kwargs_strategy,
    can_point_transform,
    chart_classes,
    chart_init_kwargs,
    chart_time_chain,
    charts,
    charts_like,
    product_charts,
)
from ._src.distances import distance_moduli, distances, parallaxes
from ._src.vectors import (
    coord_roles,
    physical_roles,
    point_role,
    pointedvectors,
    roles,
    vectors,
    vectors_with_target_chart,
)
