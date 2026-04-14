r"""Curve-attached reference frames for $\tau$-parameterised curves.

This package provides **Frenet--Serret** and **Bishop** (rotation-minimising)
curve-attached reference frames that integrate with the ``coordinax.frames``
frame-transition system.

Public API
----------
.. autosummary::

    AbstractParallelTransportFrame
    AbstractParallelTransportTransform
    FrenetSerretTransform
    FrenetSerretFrame
    BishopTransform
    BishopFrame

Typical usage::

    import coordinax.curveframes as cxfc

    fs_frame = cxfc.FrenetSerretFrame.from_curve(base_frame, curve)
    b_frame  = cxfc.BishopFrame.from_curve(base_frame, curve)

See Also
--------
coordinax.frames : The frame-transition dispatch system.
coordinax.transforms : Transform primitives (Translate, Rotate, etc.).
"""

__all__ = (
    "AbstractParallelTransportFrame",
    "AbstractParallelTransportTransform",
    "BishopFrame",
    "BishopTransform",
    "FrenetSerretFrame",
    "FrenetSerretTransform",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.curveframes"):
    from ._src import (
        AbstractParallelTransportFrame,
        AbstractParallelTransportTransform,
        BishopFrame,
        BishopTransform,
        FrenetSerretFrame,
        FrenetSerretTransform,
    )

del install_import_hook
