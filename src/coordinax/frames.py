"""Reference frames and transformations between them."""

from ._src.frames.api import frame_transform_op
from ._src.frames.base import AbstractReferenceFrame
from ._src.frames.errors import FrameTransformError

# Space frames
# isort: split
from . import _coordinax_space_frames
from ._coordinax_space_frames import *  # noqa: F403

__all__ = [
    "frame_transform_op",
    "AbstractReferenceFrame",
    "FrameTransformError",
]
__all__ += _coordinax_space_frames.__all__

# clean up namespace
del _coordinax_space_frames
