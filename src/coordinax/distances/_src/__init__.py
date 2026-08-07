"""`coordinax.distances` private module.

Note that this module is private. Users should use the public API.

This module depends on the following modules:

- utils & typing

"""

from .base import *
from .measures import *
from .register_converters import *
from .register_primitives import *
from .register_unxt import *
from coordinax._src.optional_deps import OptDeps

if OptDeps.UNXTS_PARAMETRIC.installed:
    from .register_parametric import *
