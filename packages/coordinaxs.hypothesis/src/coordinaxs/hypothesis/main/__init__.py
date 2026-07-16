"""Core classes and functions for coordinaxs.hypothesis."""

__all__ = (
    # Angles
    "angles",
    # Distances
    "distances",
    # Charts
    "chart_classes",
    "chart_init_kwargs",
    "charts",
    "charts_like",
    "cdicts",
    # Manifolds
    "atlas_classes",
    "atlases",
    "manifold_classes",
    "manifolds",
    # Representations
    "geometry_classes",
    "geometries",
    "basis_classes",
    "bases",
    "semantic_classes",
    "semantics",
    "valid_basis_classes_for_geometry",
    "valid_semantic_classes_for_geometry",
    "representations",
)

from coordinaxs.hypothesis.angles import angles
from coordinaxs.hypothesis.charts import (
    cdicts,
    chart_classes,
    chart_init_kwargs,
    charts,
    charts_like,
)
from coordinaxs.hypothesis.distances import distances
from coordinaxs.hypothesis.manifolds import (
    atlas_classes,
    atlases,
    manifold_classes,
    manifolds,
)
from coordinaxs.hypothesis.representations import (
    bases,
    basis_classes,
    geometries,
    geometry_classes,
    representations,
    semantic_classes,
    semantics,
    valid_basis_classes_for_geometry,
    valid_semantic_classes_for_geometry,
)
