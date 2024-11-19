"""Frames sub-package."""

__all__ = ["frame_transform_op"]

from typing import Annotated as Antd, Any
from typing_extensions import Doc

from plum import dispatch


@dispatch.abstract  # type: ignore[misc]
def frame_transform_op(
    from_frame: Antd[Any, Doc("frame to convert from")],
    to_frame: Antd[Any, Doc("frame to convert to")],
    /,
) -> Any:
    """Make a frame transform."""
