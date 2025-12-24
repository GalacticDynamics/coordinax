"""`coordinax.transforms ` Module."""

__all__ = (
    "point_transform",
    "physical_tangent_transform",
    "frame_cart",
    "cartesian_chart",
    "apply_op",
    "pushforward",
    "pullback",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.transforms"):
    # Import dispatch registrations (side effect: registers plum dispatches)
    from ._src import transformations as _transformations  # noqa: F401
    from ._src.api import (
        apply_op,
        cartesian_chart,
        frame_cart,
        physical_tangent_transform,
        point_transform,
        pullback,
        pushforward,
    )


del install_import_hook, RUNTIME_TYPECHECKER
