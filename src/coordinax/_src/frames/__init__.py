"""Frames sub-package.

This is the private implementation of the frames sub-package.

"""

__all__ = [
    # Frames
    "AbstractReferenceFrame",
    "FrameTransformError",
    "NoFrame",
    "TransformedReferenceFrame",
    "frame_transform_op",
    "frame_of",
    # Example frames
    "Alice",
    "FriendOfAlice",
    "Bob",
    # Coordinate
    "AbstractCoordinate",
    "Coordinate",
]

from .api import frame_of, frame_transform_op
from .base import AbstractReferenceFrame
from .coordinate import AbstractCoordinate, Coordinate
from .errors import FrameTransformError
from .example import Alice, Bob, FriendOfAlice
from .null import NoFrame
from .xfm import TransformedReferenceFrame

# isort: split
from . import (
    register_frames,  # noqa: F401
    register_ops,  # noqa: F401
    register_primitives,  # noqa: F401
    register_vectorapi,  # noqa: F401
)
