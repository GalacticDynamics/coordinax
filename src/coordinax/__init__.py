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
    "equivalent",
)

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
from coordinax.vectors import (
    Coordinate,
    Point,
    Tangent,
    ToUnitsOptions,
    equivalent,
)

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
# not yet resolvable, so the interop import fails.
#
# Because interop is an *optional* extra, a failed entry-point *load* must never
# break `import coordinax`. Rather than classifying failures, the loader records
# each load failure (in `_OPTIONAL_INTEROP_STATE["failed"]`) and does not
# re-raise it. A transient cyclic failure recovers on a later call — packages
# that participate in such a cycle re-invoke this once their own symbols exist
# (see `coordinaxs.astro.__init__`), and the retry succeeds and clears the
# failure. A genuine failure simply stays recorded. This needs no inspection of
# import-machinery state and behaves identically in every import order.
# (Errors from entry-point *discovery* itself — e.g. corrupt distribution
# metadata — indicate a broken environment and are left to propagate.)

_INTEROP_ENTRYPOINT_GROUP: Final = "coordinaxs.interop"
#: Pre-rename group name, still honoured so third-party interop
#: distributions published against it are not silently dropped.
_LEGACY_INTEROP_ENTRYPOINT_GROUP: Final = "coordinax.interop"
#: ``loaded``: names of entry points whose module imported successfully.
#: ``failed``: name -> the last exception raised while loading it (retryable;
#: cleared on a later successful load). Inspect ``failed`` to diagnose an
#: installed-but-broken interop.
_OPTIONAL_INTEROP_STATE: dict[str, Any] = {
    "loading": False,
    "loaded": set(),
    "failed": {},
}


def _load_optional_interop() -> None:
    """Import interop packages registered in the ``coordinaxs.interop`` group.

    Interop distributions register their conversions and chart transitions as an
    import side effect. Because interop is an *optional* extra, an entry point
    that fails to *load* must never break ``import coordinax``: each entry point
    is loaded at most once, and a load that raises is recorded in
    ``_OPTIONAL_INTEROP_STATE["failed"]`` (with its traceback cleared) and retried
    on the next call rather than propagated. Errors from entry-point *discovery*
    itself — e.g. corrupt distribution metadata — are a broken environment and
    are left to propagate.

    Retryability is what makes registration import-order independent. Interop
    participates in a cycle — ``coordinaxs.interop.astropy`` references
    ``coordinaxs.astro`` types, and ``coordinaxs.astro`` imports
    ``coordinax.frames``, which (now that ``coordinax`` is a regular package)
    runs this module. When the sibling is imported first, this loader runs while
    the sibling is only partially initialized, so the interop import fails; it is
    recorded and simply *succeeds on the retry* once the sibling finishes. No
    inspection of import-machinery state is needed, and the behaviour is the same
    whichever module is imported first.

    Contract for interop authors: a failed/pending entry point is retried only
    when something calls this again. Core calls it once at the end of its own
    import, and ``coordinaxs.astro`` re-invokes it at the end of *its* import. An
    interop that participates in a cycle through a *different* sibling should
    have that sibling likewise call ``coordinax._load_optional_interop()`` at the
    end of its ``__init__`` (once its public symbols exist), or a sibling-first
    import leaves the interop recorded in ``failed`` and unregistered until the
    next call. A genuinely broken interop stays in ``failed`` in every ordering.
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
            except Exception as exc:  # noqa: BLE001
                # Optional extra: a failed load must never break `import
                # coordinax`. Record the failure (retryable) instead of raising.
                # Clear the traceback first: this dict lives for the process
                # lifetime, and a retained traceback pins its stack frames and
                # their locals. A transient cyclic failure recovers on a later
                # call; a genuine one persists here.
                state["failed"][ep.name] = exc.with_traceback(None)
            else:
                state["loaded"].add(ep.name)
                state["failed"].pop(ep.name, None)
    finally:
        state["loading"] = False


_load_optional_interop()
