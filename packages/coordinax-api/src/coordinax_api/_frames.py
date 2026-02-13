"""Vector API for coordinax."""

__all__ = ("frame_of", "frame_transform_op")

from typing import Any

import plum


@plum.dispatch.abstract
def frame_of(obj: Any, /) -> Any:
    """Get the frame of an object."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
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
    >>> import coordinax.charts as cxc
    >>> import coordinax.roles as cxr
    >>> import coordinax.ops as cxop
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> op
    Pipe((
      Translate(
          {'x': Q(i64[], 'km'), 'y': Q(i64[], 'km'), 'z': Q(i64[], 'km')},
          chart=Cart3D()
      ),
      Boost(
          {'x': Q(f64[], 'm / s'), 'y': Q(f64[], 'm / s'), 'z': Q(f64[], 'm / s')},
          chart=Cart3D()
      )
    ))

    Apply to a {class}`coordinax.roles.PhysVel` vector at time tau=1 year:

    >>> v = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cxr.phys_vel)
    >>> t = u.Q(1, "yr")
    >>> result = op(t, v)
    >>> print(result)
    <Vector: chart=Cart3D, role=PhysVel (x, y, z) [km / s]
        [2.698e+05 2.000e+01 3.000e+01]>

    The Translate doesn't affect PhysVel (identity), and Boost adds its velocity offset.

    """
    raise NotImplementedError  # pragma: no cover
