"""Optional dependencies. Internal use only."""

__all__ = ["HAS_ASTROPY"]

from importlib.util import find_spec

HAS_ASTROPY: bool = find_spec("astropy") is not None
