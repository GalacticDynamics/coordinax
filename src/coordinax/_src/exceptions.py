"""Exceptions for coordinax.charts module."""

__all__ = ("MismatchedManifoldError", "NoGlobalCartesianChartError")


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


class MismatchedManifoldError(Exception):
    """Raised when a manifold argument disagrees with the chart it accompanies.

    The `pt_map` rules take the manifolds explicitly as well as the charts, so
    a caller can name a manifold that is not the chart's own -- or ask for a
    transition between charts on different manifolds, which no rule implements.

    Previously a bare ``assert``. That reported the same condition, but only
    while assertions run: under ``python -O`` the guard vanishes and the call
    proceeds on a mismatched pair instead of refusing. It is also
    indistinguishable, by type, from a genuine broken invariant elsewhere in
    the call.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> try:
    ...     cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R2, cxc.cart2d)
    ... except cxc.MismatchedManifoldError as e:
    ...     print(e)
    to_M Rn(2) is not to_chart's manifold Rn(3)

    """
