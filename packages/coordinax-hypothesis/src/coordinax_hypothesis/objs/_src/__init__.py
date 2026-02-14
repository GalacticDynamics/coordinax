"""Hypothesis strategies for CsDict objects.

A CsDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CsDict objects that match
chart component schemas and role dimension requirements.
"""

__all__ = ("cdicts", "pointedvectors", "vectors", "vectors_with_target_chart")

from .cdict import cdicts
from .vectors import pointedvectors, vectors, vectors_with_target_chart
