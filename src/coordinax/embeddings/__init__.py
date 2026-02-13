"""`coordinax.embeddings` Module."""

__all__ = (
    "EmbeddedManifold",
    "AbstractEmbedded",
    "embed_point",
    "embed_tangent",
    "project_point",
    "project_tangent",
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.embeddings"):
    from ._src import AbstractEmbedded, EmbeddedManifold
    from coordinax.api import embed_point, embed_tangent, project_point, project_tangent


del setup_package
