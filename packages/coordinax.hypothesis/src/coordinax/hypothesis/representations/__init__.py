"""Hypothesis strategies for coordinax."""

__all__ = (
    # Geometry strategies
    "geometry_classes",
    "geometries",
    # Basis strategies
    "basis_classes",
    "bases",
    # Semantic strategies
    "semantic_classes",
    "semantics",
    # Representation strategies
    "valid_basis_classes_for_geometry",
    "valid_semantic_classes_for_geometry",
    "representations",
    # CDict strategies
    "cdicts",
)

from ._src import (
    bases,
    basis_classes,
    cdicts,
    geometries,
    geometry_classes,
    representations,
    semantic_classes,
    semantics,
    valid_basis_classes_for_geometry,
    valid_semantic_classes_for_geometry,
)
