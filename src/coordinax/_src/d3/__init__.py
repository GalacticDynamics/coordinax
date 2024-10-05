# pylint: disable=duplicate-code
"""3-dimensional representations."""

from . import (
    base,
    base_spherical,
    cartesian,
    compat,
    constructor,
    cylindrical,
    generic,
    lonlatspherical,
    mathspherical,
    spherical,
    transform,
)
from .base import *
from .base_spherical import *
from .cartesian import *
from .compat import *
from .constructor import *
from .cylindrical import *
from .generic import *
from .lonlatspherical import *
from .mathspherical import *
from .spherical import *
from .transform import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += cartesian.__all__
__all__ += cylindrical.__all__
__all__ += base_spherical.__all__
__all__ += spherical.__all__
__all__ += mathspherical.__all__
__all__ += lonlatspherical.__all__
__all__ += generic.__all__
__all__ += transform.__all__
__all__ += compat.__all__
__all__ += constructor.__all__
