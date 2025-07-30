"""Base implementation of coordinate frames."""

__all__ = ["AbstractReferenceFrame"]

from collections.abc import Mapping
from typing import Annotated as Antd, Any
from typing_extensions import Doc

import equinox as eqx
from plum import dispatch

from .api import frame_transform_op
from coordinax._src.operators import AbstractOperator


class AbstractReferenceFrame(eqx.Module):
    """Base class for all reference frames."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractReferenceFrame]", obj: Any, /
    ) -> "AbstractReferenceFrame":
        """Construct a reference frame."""
        raise NotImplementedError  # pragma: no cover

    # ---------------------------------------------------------------
    # Transformations

    # TODO: rename?
    def transform_op(
        self, to_frame: Antd["AbstractReferenceFrame", Doc("frame to transform to")], /
    ) -> Antd[AbstractOperator, Doc("frame transform operator")]:
        """Make a frame transform operator.

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
        return frame_transform_op(self, to_frame)


# =============================================================================
# Constructors


@AbstractReferenceFrame.from_.dispatch
def from_(
    cls: type[AbstractReferenceFrame], obj: Mapping[str, Any], /
) -> AbstractReferenceFrame:
    """Construct a reference frame from a mapping.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice.from_({})
    >>> alice
    Alice()

    >>> bob = cxf.Bob.from_({})
    >>> print(bob)
    Bob()

    """
    return cls(**obj)


@AbstractReferenceFrame.from_.dispatch
def from_(
    cls: type[AbstractReferenceFrame], obj: AbstractReferenceFrame, /
) -> AbstractReferenceFrame:
    """Construct a reference frame from another reference frame.

    Raises
    ------
    TypeError
        If the input object is not a subclass of the target class.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> cxf.AbstractReferenceFrame.from_(alice) is alice
    True

    >>> try:
    ...     cxf.Bob.from_(alice)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct 'Bob' from Alice()

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls.__qualname__!r} from {obj}"
        raise TypeError(msg)

    return obj
