"""`coordinax.transforms ` Module."""

__all__ = (
    "point_transform",
    "physical_tangent_transform",
    "coord_transform",
    "frame_cart",
    "cartesian_chart",
    "apply_op",
    "pushforward",
    "pullback",
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.transforms"):
    # Import dispatch registrations (side effect: registers plum dispatches)
    from ._src import *
    from coordinax.api import (
        apply_op,
        cartesian_chart,
        coord_transform,
        frame_cart,
        physical_tangent_transform,
        point_transform,
        pullback,
        pushforward,
    )


del setup_package
