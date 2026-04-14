"""``import coordinax.core as cx``."""

__all__ = (  # distances
    "Distance",
    # angles
    "Angle",
)

from coordinax.angles import Angle
from coordinax.distances import Distance

try:  # noqa: SIM105
    import coordinax.interop.astropy as _  # noqa: F401  # ty: ignore[unresolved-import]
except ImportError:
    pass
