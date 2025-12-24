"""Register dispatches for `unxt`."""

__all__: tuple[str, ...] = ()


from plum import dispatch

from dataclassish import replace

import coordinax.r as cxr
from .coordinate import Coordinate

# ===============================================================
# Vector conversion


@dispatch
def vconvert(target: cxr.AbstractRep, w: Coordinate, /) -> Coordinate:
    """Transform the vector representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cx.frames.NoFrame()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.vconvert(cx.Spherical3D, w)
    Coordinate(
        KinematicSpace({ 'length': Spherical3D( ... ) }),
        frame=NoFrame()
    )

    """
    return replace(w, data=w.data.vconvert(target))
