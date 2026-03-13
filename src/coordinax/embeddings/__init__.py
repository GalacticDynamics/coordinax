"""`coordinax.embeddings` Module."""

__all__ = (
    "EmbeddedManifold",
    "AbstractEmbedding",
    "EmbeddedChart",
    "embed_point",
    "embed_tangent",
    "project_point",
    "project_tangent",
    # Specific embeddings
    "TwoSphereIn3D",
    # Helper constructors
    "embedded_twosphere",
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.embeddings"):
    from ._src import (
        AbstractEmbedding,
        EmbeddedChart,
        EmbeddedManifold,
        TwoSphereIn3D,
        embedded_twosphere,
    )
    from coordinax.api.embeddings import (
        embed_point,
        embed_tangent,
        project_point,
        project_tangent,
    )


del setup_package
