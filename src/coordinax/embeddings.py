"""`coordinax.embeddings` Module."""

__all__ = (
    "EmbeddedManifold",
    "AbstractEmbedded",
    "embed_point",
    "embed_tangent",
    "project_point",
    "project_tangent",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.embeddings"):
    from ._src.api import embed_point, embed_tangent, project_point, project_tangent
    from ._src.embed import AbstractEmbedded, EmbeddedManifold


del install_import_hook, RUNTIME_TYPECHECKER
