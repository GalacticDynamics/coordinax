"""Frames sub-package."""

__all__ = ["frame_transform_op", "frame_of"]

from typing import Annotated as Antd, Any
from typing_extensions import Doc

from plum import dispatch


@dispatch.abstract
def frame_transform_op(
    from_frame: Antd[Any, Doc("frame to convert from")],
    to_frame: Antd[Any, Doc("frame to convert to")],
    /,
) -> Any:
    """Make a frame transform.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> import coordinax.frames as cxf

    >>> icrs = cxf.ICRS()
    >>> gcf = cxf.Galactocentric()

    >>> op = cxf.frame_transform_op(icrs, gcf)
    >>> op
    Pipe(( ... ))

    >>> q = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> print(op(q))
    <CartesianPos3D: (x, y, z) [kpc]
        [-11.375   1.845   0.133]>

    """


@dispatch.abstract
def frame_of(obj: Any, /) -> Any:
    """Get the frame of an object."""
