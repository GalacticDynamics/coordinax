"""Frames sub-package.

This is the private implementation of the frames sub-package.

"""

__all__ = (
    # Frames
    "AbstractReferenceFrame",
    "FrameTransformError",
    "NoFrame",
    "TransformedReferenceFrame",
    # Example frames
    "Alice",
    "FriendOfAlice",
    "Bob",
)

from .base import AbstractReferenceFrame
from .errors import FrameTransformError
from .example import Alice, Bob, FriendOfAlice
from .null import NoFrame
from .xfm import TransformedReferenceFrame

# isort: split
from . import register_frames  # noqa: F401
