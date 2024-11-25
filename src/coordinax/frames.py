"""Reference frames and transformations between them."""

from ._src.frames import (
    AbstractReferenceFrame,
    FrameTransformError,
    NoFrame,
    frame_transform_op,
)

# Register the frame transform operations
# isort: split
from ._src.frames.xfm import *  # noqa: F403

# Frames from external packages
# isort: split
from . import _coordinax_space_frames
from ._coordinax_space_frames import *  # noqa: F403

__all__ = [
    "frame_transform_op",
    "AbstractReferenceFrame",
    "NoFrame",
    "FrameTransformError",
]
__all__ += _coordinax_space_frames.__all__

# clean up namespace
del _coordinax_space_frames
