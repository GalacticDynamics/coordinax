"""``import coordinax as cx``."""

__all__ = (  # distances
    "Distance",
    # angles
    "Angle",
    # charts
    "NoGlobalCartesianChartError",
    "CartesianProductChart",
    "cartesian_chart",
    "guess_chart",
    "cdict",
    "pt_map",
    "jac_pt_map",
    "cart1d",
    "radial1d",
    "time1d",
    "cart2d",
    "polar2d",
    "cart3d",
    "cyl3d",
    "sph3d",
    "lonlat_sph3d",
    "loncoslat_sph3d",
    "math_sph3d",
    "cartnd",
    # manifolds and atlases
    "EuclideanManifold",
    "Rn",
    "FlatMetric",
    "R3",
    "S2",
    "embedded_twosphere",
    "EmbeddedChart",
    "EmbeddedManifold",
    "CustomAtlas",
    "CustomManifold",
    # frames -- frames
    "noframe",
    # frame -- transforms
    "act",
    "frame_transition",
    "simplify",
    "Composed",
    "identity",
    "Rotate",
    "Translate",
    "Scale",
    "Shear",
    # representations
    "cconvert",
    "add",
    "subtract",
    "PointGeometry",
    "point_geom",
    "TangentGeometry",
    "tangent_geom",
    "NoBasis",
    "no_basis",
    "AbstractLinearBasis",
    "CoordinateBasis",
    "coord_basis",
    "PhysicalBasis",
    "phys_basis",
    "Location",
    "loc",
    "AbstractTangentSemanticKind",
    "Displacement",
    "dpl",
    "Velocity",
    "vel",
    "Acceleration",
    "acc",
    "Representation",
    "point",
    "coord_disp",
    "coord_vel",
    "coord_acc",
    "phys_disp",
    "phys_vel",
    "phys_acc",
    "tangent_map",
    "change_basis",
    # vectors
    "Point",
    "Coordinate",
    "Tangent",
    "ToUnitsOptions",
)

import sys
import warnings
from importlib.metadata import entry_points

from typing import Any, Final

from coordinax.angles import Angle
from coordinax.charts import (
    CartesianProductChart,
    NoGlobalCartesianChartError,
    cart1d,
    cart2d,
    cart3d,
    cartesian_chart,
    cartnd,
    cdict,
    cyl3d,
    guess_chart,
    jac_pt_map,
    loncoslat_sph3d,
    lonlat_sph3d,
    math_sph3d,
    polar2d,
    pt_map,
    radial1d,
    sph3d,
    time1d,
)
from coordinax.distances import Distance
from coordinax.frames import frame_transition, noframe
from coordinax.manifolds import (
    R3,
    S2,
    CustomAtlas,
    CustomManifold,
    EmbeddedChart,
    EmbeddedManifold,
    EuclideanManifold,
    FlatMetric,
    Rn,
    embedded_twosphere,
)
from coordinax.representations import (
    AbstractLinearBasis,
    AbstractTangentSemanticKind,
    Acceleration,
    CoordinateBasis,
    Displacement,
    Location,
    NoBasis,
    PhysicalBasis,
    PointGeometry,
    Representation,
    TangentGeometry,
    Velocity,
    acc,
    add,
    cconvert,
    change_basis,
    coord_acc,
    coord_basis,
    coord_disp,
    coord_vel,
    dpl,
    loc,
    no_basis,
    phys_acc,
    phys_basis,
    phys_disp,
    phys_vel,
    point,
    point_geom,
    subtract,
    tangent_geom,
    tangent_map,
    vel,
)
from coordinax.transforms import (
    Composed,
    Rotate,
    Scale,
    Shear,
    Translate,
    act,
    identity,
    simplify,
)
from coordinax.vectors import Coordinate, Point, Tangent, ToUnitsOptions

# ============================================================================
# Optional interop registration
#
# Interop distributions register their `plum` conversions and chart transitions
# as an import side effect. They are discovered through the ``coordinaxs.interop``
# entry-point group rather than imported by name, so core never depends on its
# own optional extras: an interop package that is not installed contributes no
# entry point and is simply absent from the group.
#
# Loading is retryable because interop participates in an import cycle. An
# interop package references types from a sibling package (e.g.
# `coordinaxs.interop.astropy` uses `coordinaxs.astro.Parallax`), and that
# sibling imports `coordinax.frames`, which — now that `coordinax` is a regular
# package — runs this module. So when the sibling is imported *first*, this
# loader runs while the sibling is only partially initialized and its types are
# not yet resolvable. Rather than pre-guessing that state, the loader attempts
# the import and classifies the failure: if any `coordinaxs.*` module is still
# executing its body, the entry point is left pending for a later call;
# otherwise the failure is real and propagates. Packages that participate in
# such a cycle re-invoke this once their own symbols exist (see
# `coordinaxs.astro.__init__`), which is what completes the registration.

_INTEROP_ENTRYPOINT_GROUP: Final = "coordinaxs.interop"
#: Pre-rename group name, still honoured so third-party interop
#: distributions published against it are not silently dropped.
_LEGACY_INTEROP_ENTRYPOINT_GROUP: Final = "coordinax.interop"
_OPTIONAL_INTEROP_STATE: dict[str, Any] = {"loading": False, "loaded": set()}


def _coordinaxs_is_initializing() -> bool:
    """Whether any ``coordinaxs`` module is still executing its module body."""
    for name in list(sys.modules):
        if name != "coordinaxs" and not name.startswith("coordinaxs."):
            continue
        spec = getattr(sys.modules.get(name), "__spec__", None)
        if getattr(spec, "_initializing", False):
            return True
    return False


def _load_optional_interop() -> None:
    """Import interop packages registered in the ``coordinaxs.interop`` group.

    Idempotent and retryable: each entry point is loaded at most once, and any
    that cannot be loaded yet (because a package it references is mid-import)
    is left pending for a later call.

    Contract for interop authors: a pending entry point is only retried when
    something calls this function again. Core calls it once at the end of its
    own import, and ``coordinaxs.astro`` re-invokes it at the end of *its* import
    (so astro-first ordering works). If you write an interop that participates
    in an import cycle through a *different* sibling package, that sibling must
    likewise call ``coordinax._load_optional_interop()`` at the end of its
    ``__init__`` once its public symbols exist — otherwise a sibling-first import
    leaves your interop permanently pending and silently unregistered.

    Known limitation: a genuinely broken interop package whose import fails
    *while* some ``coordinaxs`` module is still initializing is indistinguishable
    from the expected cycle, so it is left pending rather than raised. In
    practice that means a broken interop is reported when `coordinax` is
    imported first (the common case, and the documented entry point) but is
    silently skipped when the interop's sibling package is imported first.
    """
    state = _OPTIONAL_INTEROP_STATE
    if state["loading"]:  # guard against re-entrant loading
        return

    state["loading"] = True
    try:
        current = list(entry_points(group=_INTEROP_ENTRYPOINT_GROUP))
        seen = {ep.name for ep in current}
        legacy = [
            ep
            for ep in entry_points(group=_LEGACY_INTEROP_ENTRYPOINT_GROUP)
            if ep.name not in seen
        ]
        if legacy:
            names = ", ".join(sorted(ep.name for ep in legacy))
            warnings.warn(
                f"Entry point(s) {names} register interop under the legacy "
                f"'{_LEGACY_INTEROP_ENTRYPOINT_GROUP}' group. That group is "
                f"deprecated; publish under '{_INTEROP_ENTRYPOINT_GROUP}' "
                "instead. Support for the legacy group will be removed in a "
                "future release.",
                DeprecationWarning,
                stacklevel=3,
            )
        eps = sorted(current + legacy, key=lambda e: e.name)
        for ep in eps:
            if ep.name in state["loaded"]:
                continue
            try:
                ep.load()  # importing the module performs the registration
            except Exception:
                # Only the known import cycle is tolerated; leave this entry
                # point pending so a later call retries it. Anything else is a
                # genuine failure in an installed interop package.
                if not _coordinaxs_is_initializing():
                    raise
            else:
                state["loaded"].add(ep.name)
    finally:
        state["loading"] = False


_load_optional_interop()
