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

    >>> op = cxf.frame_transform_op(cxf.Alice(), cxf.Bob())
    >>> op
    Pipe(( ... ))

    >>> t = u.Quantity(2.5, "yr")
    >>> q = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> _, q_bob = op(t, q)
    >>> print(q_bob)
    <CartesianPos3D: (x, y, z) [kpc]
        [1.001 2.    3.   ]>

    """


@dispatch.abstract
def frame_of(obj: Any, /) -> Any:
    """Get the frame of an object."""
