r"""Transform operators and transformation-group markers.

Examples
--------
>>> import unxt as u
>>> import coordinax.transforms as cxfm

>>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> op
Rotate(f64[3,3](jax))

"""

import warnings
from importlib.metadata import entry_points

from typing import Final

from coordinax._src.optional_exports import load_exports
from coordinax._src.setup_package import install_import_hook

__all__: tuple[str, ...] = (
    # API
    "act",
    "act_jet",
    "pushforward",
    "simplify",
    "compose",
    "evaluate_at",
    "is_time_dependent",
    "tau_derivative",
    # Transformations
    "AbstractTransform",
    "AbstractCompositeTransform",
    "Boost",
    "Identity",
    "Composed",
    "Translate",
    "LorentzBoost",
    "Affine",
    "Linear",
    "Rotate",
    "Reflect",
    "Scale",
    "Shear",
    "TimeDep",
    "identity",
    # Sub-namespaces
    "builders",
    "groups",
)

with install_import_hook("coordinax.transforms"):
    from . import builders, groups
    from ._src.actions import (
        AbstractCompositeTransform,
        AbstractTransform,
        Affine,
        Boost,
        Composed,
        Identity,
        Linear,
        LorentzBoost,
        Reflect,
        Rotate,
        Scale,
        Shear,
        TimeDep,
        Translate,
        evaluate_at,
        identity,
        is_time_dependent,
        tau_derivative,
    )
    from coordinaxs.api.transforms import act, act_jet, compose, pushforward, simplify


# Extension point: distributions may register transform symbols under the
# ``coordinaxs.transforms`` entry-point group. No in-tree distribution
# currently registers here; the consumer is kept live for downstream packages.
_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinaxs.transforms"
#: Pre-rename group name, still honoured (with a deprecation warning) so
#: third-party registrants published against it are not silently dropped.
_LEGACY_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinax.transforms"
_OPTIONAL_TRANSFORM_EXPORTS_STATE: dict[str, bool] = {"loading": False}


def _load_optional_transform_exports() -> None:
    """Load optional transform symbols.

    ``coordinaxs.transforms`` entry-point group.
    """
    if _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"]:
        return

    _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = True
    try:
        current = list(entry_points(group=_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP))
        seen = {ep.name for ep in current}
        legacy = [
            ep
            for ep in entry_points(group=_LEGACY_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP)
            if ep.name not in seen
        ]
        if legacy:
            names = ", ".join(sorted(ep.name for ep in legacy))
            warnings.warn(
                f"Entry point(s) {names} register transform symbols under the "
                f"legacy '{_LEGACY_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' group. "
                f"That group is deprecated; publish under "
                f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' instead. Support for "
                "the legacy group will be removed in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )
        eps = sorted(current + legacy, key=lambda ep: ep.name)
        exported = load_exports(
            eps, group=_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP, noun="transform export"
        )
        globals().update(exported)
    finally:
        _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = False


_load_optional_transform_exports()

del (install_import_hook, Final)
