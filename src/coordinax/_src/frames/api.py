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
