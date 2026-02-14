"""Abstract dispatch API for `coordinax`.

This package defines the abstract dispatch interfaces for `coordinax`'s core
functionality.
"""

__all__ = (
    "__version__",
    # charts
    "cartesian_chart",
    "guess_chart",
    # embeddings
    "embed_point",
    "project_point",
    "embed_tangent",
    "project_tangent",
    # metrics
    "metric_of",
    # frames
    "frame_of",
    "frame_transform_op",
    # roles
    "as_disp",
    "guess_role",
    # transformations
    "physicalize",
    "coordinateize",
    "point_transform",
    "physical_tangent_transform",
    "coord_transform",
    "frame_cart",
    "pushforward",
    "pullback",
    # operators
    "apply_op",
    "simplify",
    # objects
    "vconvert",
    "cdict",
    # custom types
    "ComponentKey",
    "ProductComponentKey",
    "ComponentsKey",
    "CDict",
    "CsDict",
)

from ._charts import cartesian_chart, cdict, guess_chart
from ._custom_types import (
    CDict,
    ComponentKey,
    ComponentsKey,
    CsDict,
    ProductComponentKey,
)
from ._embeddings import embed_point, embed_tangent, project_point, project_tangent
from ._frames import frame_of, frame_transform_op
from ._metrics import metric_of
from ._objs import vconvert
from ._operators import apply_op, simplify
from ._roles import as_disp, guess_role
from ._transformations import (
    coord_transform,
    coordinateize,
    frame_cart,
    physical_tangent_transform,
    physicalize,
    point_transform,
    pullback,
    pushforward,
)
from ._version import version as __version__
