"""Frames sub-package."""

__all__ = ["frame_transform_op", "frame_of"]

from typing import Any

from plum import dispatch


@dispatch.abstract
def frame_transform_op(from_frame: Any, to_frame: Any, /) -> Any:
    """Make a frame transform.

    Parameters
    ----------
    from_frame : AbstractReferenceFrame
        The reference frame to transform from.
    to_frame : AbstractReferenceFrame
        The reference frame to transform to.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> op
    Pipe(( ... ))

    >>> q = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(1, "yr")
    >>> print(op(t, q)[1])
    <CartesianPos3D: (x, y, z) [kpc]
        [1. 2. 3.]>

    """


@dispatch.abstract
def frame_of(obj: Any, /) -> Any:
    """Get the frame of an object."""
