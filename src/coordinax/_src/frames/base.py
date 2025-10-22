"""Base implementation of coordinate frames."""

__all__ = ["AbstractReferenceFrame"]

from collections.abc import Mapping
from typing import Any

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
    def transform_op(self, to_frame: "AbstractReferenceFrame", /) -> AbstractOperator:
        """Make a frame transform operator.

        Parameters
        ----------
        to_frame : AbstractReferenceFrame
            The reference frame to transform to.

        Returns
        -------
        AbstractOperator
            The operator that transforms coordinates from this frame to
            `to_frame`.

        Examples
        --------
        >>> import coordinax.frames as cxf

        >>> alice = cxf.Alice()
        >>> bob = cxf.Bob()
        >>> op = alice.transform_op(bob)
        >>> op
        Pipe(( ... ))

        >>> op = bob.transform_op(alice)
        >>> op
        Pipe(( ... ))

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

    >>> icrs = cxf.ICRS.from_({})
    >>> icrs
    ICRS()

    >>> gcf = cxf.Galactocentric.from_({})
    >>> print(gcf)
    Galactocentric( ... )

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
    ...     cxf.Galactocentric.from_(alice)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct 'Galactocentric' from Alice()

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls.__qualname__!r} from {obj}"
        raise TypeError(msg)

    return obj
