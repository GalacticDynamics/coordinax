"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps",)

from typing import final

from optional_dependencies import OptionalDependencyEnum, auto


@final
class OptDeps(OptionalDependencyEnum):  # pylint: disable=invalid-enum-extension
    """Optional dependencies for ``coordinax``."""

    COORDINAX_INTEROP_ASTROPY = auto()
