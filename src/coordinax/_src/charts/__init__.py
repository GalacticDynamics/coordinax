"""Chart classes and function dispatches.

This module defines the built-in chart classes and function dispatches for
chart-related operations. The actual chart instances are not defined here, since
they depend on the manifold, and so are defined in the manifold-specific
submodules (e.g. ``.euclidean``).

"""

from .d0 import *
from .d1 import *
from .d2 import *
from .d3 import *
from .d6 import *
from .dn import *
from .jacobian import *
from .register_cdict import *
from .register_guess import *
from .register_ptmap import *
