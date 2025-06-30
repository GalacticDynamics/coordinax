"""Astronomy reference frames."""

__all__ = ["AbstractSpaceFrame"]


import coordinax.frames as cxf


class AbstractSpaceFrame(cxf.AbstractReferenceFrame):
    """ABC for space-related reference frames."""
