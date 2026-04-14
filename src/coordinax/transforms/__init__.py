r"""Transform operators and transformation-group markers.

Examples
--------
>>> import unxt as u
>>> import coordinax.transforms as cxfm

>>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> op
Rotate(f64[3,3](jax))

"""

from importlib.metadata import entry_points

from collections.abc import Mapping
from typing import Final

from ._setup_package import install_import_hook

__all__: tuple[str, ...] = (
    # API
    "act",
    "simplify",
    "compose",
    "materialize_transform",
    # Groups
    "AbstractTransformGroup",
    "IdentityGroup",
    "DiffeomorphismGroup",
    "AffineGroup",
    "EuclideanGroup",
    "OrthogonalGroup",
    "SpecialOrthogonalGroup",
    "PoincareGroup",
    "LorentzGroup",
    "ProperOrthochronousLorentzGroup",
    # Transformations
    "AbstractTransform",
    "AbstractCompositeTransform",
    "Boost",
    "Identity",
    "Composed",
    "Translate",
    "Rotate",
    "Reflect",
    "Scale",
    "Shear",
    "identity",
)

with install_import_hook("coordinax.transforms"):
    from ._src.actions import (
        AbstractCompositeTransform,
        AbstractTransform,
        Boost,
        Composed,
        Identity,
        Reflect,
        Rotate,
        Scale,
        Shear,
        Translate,
        identity,
        materialize_transform,
    )
    from ._src.groups import (
        AbstractTransformGroup,
        AffineGroup,
        DiffeomorphismGroup,
        EuclideanGroup,
        IdentityGroup,
        LorentzGroup,
        OrthogonalGroup,
        PoincareGroup,
        ProperOrthochronousLorentzGroup,
        SpecialOrthogonalGroup,
    )
    from coordinax.api.transforms import act, compose, simplify


_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinax.transforms"
_OPTIONAL_TRANSFORM_EXPORTS_STATE: dict[str, bool] = {"loading": False}


def _load_optional_transform_exports() -> None:
    """Load optional transform symbols.

    ``coordinax.transforms`` entry-point group.
    """
    if _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"]:
        return

    _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = True
    exported: dict[str, object] = {}
    export_owners: dict[str, str] = {}

    try:
        eps = sorted(
            entry_points(group=_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP),
            key=lambda ep: ep.name,
        )
        for ep in eps:
            provider = ep.load()
            if not callable(provider):
                msg = (
                    f"Entry point {ep.name!r} in group "
                    f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' "
                    "is not callable."
                )
                raise TypeError(msg)
            exports = provider()
            if not isinstance(exports, Mapping):
                msg = (
                    f"Entry point {ep.name!r} in group "
                    f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' "
                    "must return a mapping."
                )
                raise TypeError(msg)
            for name, value in exports.items():
                if not isinstance(name, str):
                    msg = (
                        f"Entry point {ep.name!r} in group "
                        f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' produced "
                        "a non-string export name."
                    )
                    raise TypeError(msg)
                if name in exported and exported[name] is not value:
                    msg = (
                        f"Conflicting transform export {name!r} from entry points "
                        f"{export_owners[name]!r} and {ep.name!r}."
                    )
                    raise RuntimeError(msg)
                exported[name] = value
                export_owners[name] = ep.name

        globals().update(exported)
    finally:
        _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = False


_load_optional_transform_exports()

del (
    install_import_hook,
    Final,
)
