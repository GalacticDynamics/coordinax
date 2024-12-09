"""Reference frames and transformations between them."""

__all__ = [
    "AbstractReferenceFrame",
    "FrameTransformError",
    "NoFrame",
    "frame_transform_op",
]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.frames", RUNTIME_TYPECHECKER):
    from ._src.frames import (
        AbstractReferenceFrame,
        FrameTransformError,
        NoFrame,
        frame_transform_op,
    )

    # Register the frame transform operations
    # isort: split
    from ._src.frames.register_transforms import *  # noqa: F403

    # Frames from external packages
    # isort: split
    from . import _coordinax_space_frames
    from ._coordinax_space_frames import *  # noqa: F403


__all__ += _coordinax_space_frames.__all__

# clean up namespace
del _coordinax_space_frames, RUNTIME_TYPECHECKER, install_import_hook
