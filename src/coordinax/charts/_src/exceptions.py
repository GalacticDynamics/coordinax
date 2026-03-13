"""Exceptions for coordinax.charts module."""

__all__ = ("NoGlobalCartesianChartError",)


class NoGlobalCartesianChartError(Exception):
    """Raised when a chart has no global Cartesian representation.

    Some charts represent coordinates on curved manifolds (e.g., 2-sphere)
    that cannot be globally mapped to a flat Cartesian space without
    singularities or discontinuities.

    Examples
    --------
    2-sphere charts (intrinsic coordinates on a spherical surface) have no
    global Cartesian 2D representation. To work with these charts:

    - Use an ``EmbeddedChart`` to embed in 3D Euclidean space
    - Use local projections when available
    - Work directly in the intrinsic coordinates

    """
