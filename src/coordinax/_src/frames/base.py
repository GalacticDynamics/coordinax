"""Base implementation of coordinate frames."""

__all__ = ["AbstractReferenceFrame"]

from collections.abc import Mapping
from typing import Annotated as Antd, Any
from typing_extensions import Doc

import equinox as eqx
from plum import dispatch

from .api import frame_transform_op
from coordinax._src.operators.base import AbstractOperator


class AbstractReferenceFrame(eqx.Module):  # type: ignore[misc]
    """Base class for all reference frames."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractReferenceFrame]", obj: Any, /
    ) -> "AbstractReferenceFrame":
        """Construct a reference frame."""

    # TODO: examples
    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractReferenceFrame]", obj: Mapping[str, Any], /
    ) -> "AbstractReferenceFrame":
        """Construct a reference frame from a mapping."""
        return cls(**obj)

    # ---------------------------------------------------------------
    # Transformations

    # TODO: rename?
    def transform_op(
        self, to_frame: Antd["AbstractReferenceFrame", Doc("frame to transform to")], /
    ) -> Antd[AbstractOperator, Doc("frame transform operator")]:
        """Make a frame transform operator.

        Examples
        --------
        >>> import coordinax.frames as cxf

        >>> icrs = cxf.ICRS()
        >>> gcf = cxf.Galactocentric()
        >>> op = icrs.transform_op(gcf)  # ICRS to Galactocentric
        >>> op
        _ICRS2GCFOperator( gcf=Galactocentric( ... ) )

        >>> op = gcf.transform_op(icrs)  # Galactocentric to ICRS
        >>> op
        _GCF2ICRSOperator( gcf=Galactocentric( ... ) )

        """
        return frame_transform_op(self, to_frame)
