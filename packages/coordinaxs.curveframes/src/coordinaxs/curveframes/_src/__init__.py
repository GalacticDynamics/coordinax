"""Internal implementation package for ``coordinaxs.curveframes``.

This sub-package re-exports all public symbols from the individual
implementation modules:

- {mod}`.base` — abstract base classes.
- {mod}`.frenetserret` — Frenet--Serret transform and frame.
- {mod}`.bishop` — Bishop (rotation-minimising) transform and frame.
- {mod}`.chart` — `TubularChart`, on the parameterized branch.
- {mod}`.register_frames` — ``frame_transition`` dispatch registrations.
- {mod}`.register_ptmap` — ``pt_map`` dispatch registrations for `TubularChart`.
"""

from . import register_ptmap  # noqa: F401
from .base import *
from .bishop import *
from .chart import *
from .frenetserret import *
from .register_frames import *
