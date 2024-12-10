"""Representation of coordinates in different systems."""

__all__ = ["IrreversibleDimensionChange"]


class IrreversibleDimensionChange(UserWarning):
    """Raised when a dimension change is irreversible.

    This exception is raised when a dimension change is irreversible.
    For example, changing from Cartesian3D to a Cartesian2D is irreversible
    because the z-component is lost.

    Examples
    --------
    >>> import warnings
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     _ = vec.vconvert(cx.vecs.CartesianPos2D)
    >>> print(w[0].message)
    irreversible dimension change

    """
