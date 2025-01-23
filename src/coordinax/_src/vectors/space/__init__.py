"""Space."""

__all__ = ["Space"]

from .core import Space

# Register by import
# isort: split
from . import register_dataclassish  # noqa: F401
