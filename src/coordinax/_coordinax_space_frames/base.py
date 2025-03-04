"""Astronomy reference frames."""

__all__ = ["AbstractSpaceFrame"]


from coordinax._src.frames import AbstractReferenceFrame


class AbstractSpaceFrame(AbstractReferenceFrame):
    """ABC for space-related reference frames."""
