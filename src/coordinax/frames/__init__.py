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
        {'x': Q(i64[], 'm'), 'y': Q(i64[], 'm'), 'z': Q(i64[], 'm')}, chart=Cart3D()
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

from importlib.metadata import entry_points

from collections.abc import Mapping
from typing import Final

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
)

with install_import_hook("coordinax.frames"):
    from ._src import (
        AbstractReferenceFrame,
        AbstractTransformedReferenceFrame,
        Alex,
        Alice,
        FrameTransformError,
        NoFrame,
        TransformedReferenceFrame,
        alex,
        alice,
        noframe,
    )
    from coordinax.api.frames import frame_transition


_FRAME_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinax.frames"
_OPTIONAL_FRAME_EXPORTS_STATE: dict[str, bool] = {"loading": False}


def _load_optional_frame_exports() -> None:
    """Load optional frame symbols from the ``coordinax.frames`` entry-point group."""
    # Guard against recursive entry-point loading during import-time cycles.
    if _OPTIONAL_FRAME_EXPORTS_STATE["loading"]:
        return

    _OPTIONAL_FRAME_EXPORTS_STATE["loading"] = True
    exported: dict[str, object] = {}
    export_owners: dict[str, str] = {}

    try:
        entrypoints = sorted(
            entry_points(group=_FRAME_EXPORTS_ENTRYPOINT_GROUP), key=lambda ep: ep.name
        )
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
