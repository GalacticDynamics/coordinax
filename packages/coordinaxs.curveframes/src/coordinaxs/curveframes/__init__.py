r"""Curve-attached reference frames for $\tau$-parameterised curves.

This package provides **Frenet--Serret** and **Bishop** (rotation-minimising)
curve-attached reference frames that integrate with the ``coordinax.frames``
frame-transition system.

Public API
----------
.. autosummary::

    AbstractCurveFrameBuilder
    AbstractParallelTransportFrame
    ArcLength
    AtTime
    LagrangianArcLength
    FrenetSerretBuilder
    FrenetSerretFrame
    BishopBuilder
    BishopFrame
    TubularChart
    nearest_tau

Typical usage::

    import coordinaxs.curveframes as cxfc

    fs_frame = cxfc.FrenetSerretFrame.from_curve(base_frame, curve, tau_unit)
    b_frame  = cxfc.BishopFrame.from_curve(base_frame, curve, tau_unit)

See Also
--------
coordinax.frames : The frame-transition dispatch system.
coordinax.transforms : Transform primitives (Translate, Rotate, etc.).

"""

__all__ = (
    "AbstractCurveFrameBuilder",
    "AbstractParallelTransportFrame",
    "ArcLength",
    "AtTime",
    "BishopBuilder",
    "BishopFrame",
    "FrenetSerretBuilder",
    "FrenetSerretFrame",
    "LagrangianArcLength",
    "TubularChart",
    "nearest_tau",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinaxs.curveframes"):
    from ._src import (
        AbstractCurveFrameBuilder,
        AbstractParallelTransportFrame,
        ArcLength,
        AtTime,
        BishopBuilder,
        BishopFrame,
        FrenetSerretBuilder,
        FrenetSerretFrame,
        LagrangianArcLength,
        TubularChart,
        nearest_tau,
    )

del install_import_hook
