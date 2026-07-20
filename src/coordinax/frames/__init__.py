r"""Reference frames and transformations between them.

Examples
--------
>>> import quaxed.numpy as jnp
>>> import unxt as u
>>> import coordinax.vectors as cxv
>>> import coordinax.frames as cxf

Let's transform a position from Alice's frame to Alex's frame:

>>> op = cxf.frame_transition(cxf.alice, cxf.alex)
>>> op
Composed((
    Translate(
        {'x': Q(i64[], 'm'), 'y': Q(i64[], 'm'), 'z': Q(i64[], 'm')},
        chart=Cart3D(M=Rn(3))
    ),
    Rotate(f64[3,3](jax))
))

>>> q_alice = cxv.Point.from_([0, 0, 0], "km")
>>> t = u.Q(2.5, "yr")
>>> q_alex = op(t, q_alice)
>>> print(q_alex.round(3))
<Point: chart=Cart3D (x, y, z) [km]
    [0.   0.01 0.  ]>

Now let's create a new transformed frame and work with it:

>>> import coordinax.transforms as cxfm
>>> R = cxfm.Rotate([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
>>> frame = cxf.TransformedReferenceFrame(cxf.alice, R)
>>> frame
TransformedReferenceFrame(base_frame=Alice(), xop=Rotate(R=f64[3,3]))

Let's transform a position from the base frame to the transformed frame:

>>> op = cxf.frame_transition(cxf.alice, frame)

>>> q_icrs = cxv.Point.from_([1, 0, 0], "kpc")
>>> q_frame = op(q_icrs)
>>> print(q_frame)
<Point: chart=Cart3D (x, y, z) [kpc]
    [0. 1. 0.]>

>>> op.inverse(q_frame) == q_icrs
Array(True, dtype=bool)

"""

import warnings
from importlib.metadata import entry_points

from collections.abc import Mapping
from typing import Any, Final

from ._setup_package import install_import_hook

# Defined here b/c it's mutated by optional imports
__all__: tuple[str, ...] = (
    # API
    "frame_transition",
    # Reference Frames
    "AbstractReferenceFrame",
    "FrameTransformError",
    "NoFrame",
    "noframe",
    "AbstractTransformedReferenceFrame",
    "TransformedReferenceFrame",
    # Example frames
    "Alice",
    "alice",
    "Alex",
    "alex",
    "Bob",
    "bob",
)

with install_import_hook("coordinax.frames"):
    from ._src import (
        AbstractReferenceFrame,
        AbstractTransformedReferenceFrame,
        Alex,
        Alice,
        Bob,
        FrameTransformError,
        NoFrame,
        TransformedReferenceFrame,
        alex,
        alice,
        bob,
        noframe,
    )
    from coordinaxs.api.frames import frame_transition


_FRAME_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinaxs.frames"
#: Pre-rename group name. The group is a cross-distribution contract, so
#: third-party registrants published against the old name are still honoured
#: (with a deprecation warning) rather than silently dropped.
_LEGACY_FRAME_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinax.frames"
_OPTIONAL_FRAME_EXPORTS_STATE: dict[str, bool] = {"loading": False}


def _frame_export_entrypoints() -> list[Any]:
    """Entry points registering frame exports, newest group name first.

    Reads the current ``coordinaxs.frames`` group and the legacy
    ``coordinax.frames`` group. A distribution found only under the legacy name
    gets a `DeprecationWarning`; one found under both is taken from the current
    group only (no duplicate load).
    """
    current = list(entry_points(group=_FRAME_EXPORTS_ENTRYPOINT_GROUP))
    seen = {ep.name for ep in current}

    legacy = [
        ep
        for ep in entry_points(group=_LEGACY_FRAME_EXPORTS_ENTRYPOINT_GROUP)
        if ep.name not in seen
    ]
    if legacy:
        names = ", ".join(sorted(ep.name for ep in legacy))
        warnings.warn(
            f"Entry point(s) {names} register frame exports under the legacy "
            f"'{_LEGACY_FRAME_EXPORTS_ENTRYPOINT_GROUP}' group. That group is "
            f"deprecated; publish under '{_FRAME_EXPORTS_ENTRYPOINT_GROUP}' "
            "instead. Support for the legacy group will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=3,
        )

    return sorted(current + legacy, key=lambda ep: ep.name)


def _load_optional_frame_exports() -> None:
    """Load optional frame symbols from the ``coordinaxs.frames`` entry-point group."""
    # Guard against recursive entry-point loading during import-time cycles.
    if _OPTIONAL_FRAME_EXPORTS_STATE["loading"]:
        return

    _OPTIONAL_FRAME_EXPORTS_STATE["loading"] = True
    exported: dict[str, object] = {}
    export_owners: dict[str, str] = {}

    try:
        entrypoints = _frame_export_entrypoints()
        for ep in entrypoints:
            provider = ep.load()
            if not callable(provider):
                msg = (
                    f"Entry point {ep.name!r} in group "
                    f"'{_FRAME_EXPORTS_ENTRYPOINT_GROUP}' "
                    "is not callable."
                )
                raise TypeError(msg)
            exports = provider()
            if not isinstance(exports, Mapping):
                msg = (
                    f"Entry point {ep.name!r} in group "
                    f"'{_FRAME_EXPORTS_ENTRYPOINT_GROUP}' "
                    "must return a mapping."
                )
                raise TypeError(msg)
            for name, value in exports.items():
                if not isinstance(name, str):
                    msg = (
                        f"Entry point {ep.name!r} in group "
                        f"'{_FRAME_EXPORTS_ENTRYPOINT_GROUP}' produced "
                        "a non-string export name."
                    )
                    raise TypeError(msg)

                if name in exported and exported[name] is not value:
                    msg = (
                        f"Conflicting frame export {name!r} from entry points "
                        f"{export_owners[name]!r} and {ep.name!r}."
                    )
                    raise RuntimeError(msg)

                exported[name] = value
                export_owners[name] = ep.name

        globals().update(exported)
    finally:
        _OPTIONAL_FRAME_EXPORTS_STATE["loading"] = False


_load_optional_frame_exports()


# clean up namespace
del (
    install_import_hook,
    Final,
)
