"""Internal implementation package for ``coordinaxs.curveframes``.

This sub-package re-exports all public symbols from the individual
implementation modules:

- {mod}`.base` — abstract base classes.
- {mod}`.arclength` — `ArcLength`, arc-length reparametrisation of a curve.
- {mod}`.frenetserret` — Frenet--Serret transform and frame.
- {mod}`.bishop` — Bishop (rotation-minimising) transform and frame.
- {mod}`.chart` — `TubularChart`, on the parameterized branch.
- {mod}`.nearest` — `nearest_tau`, the seeded Newton solve `TubularChart`'s
  inverse `pt_map` uses.
- {mod}`.register_frames` — ``frame_transition`` dispatch registrations.
- {mod}`.register_ptmap` — ``pt_map`` dispatch registrations for `TubularChart`.

`metric_matrix` needs no registration of its own: it falls through to
`coordinax`'s generic Jacobian-pullback rule. See the "The Metric" section of
the curve-charts guide for why, and for the two builders' differing results.
"""

from .arclength import *
from .base import *
from .bishop import *
from .chart import *
from .frenetserret import *
from .nearest import *
from .register_frames import *
from .register_ptmap import *
