"""`coordinax.vecs` Module."""
# ruff:noqa: F403

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.vecs", RUNTIME_TYPECHECKER):
    from ._src import vectors
    from ._src.vectors.api import *
    from ._src.vectors.base import *
    from ._src.vectors.d1 import *
    from ._src.vectors.d2 import *
    from ._src.vectors.d3 import *
    from ._src.vectors.d4 import *
    from ._src.vectors.dn import *
    from ._src.vectors.exceptions import *
    from ._src.vectors.space import *

    # Register vector transformations, functions, etc.
    # isort: split
    from ._src.vectors import funcs, transform

    # Interoperability
    # isort: split
    from ._src.vectors import compat

__all__: list[str] = []
__all__ += vectors.api.__all__
__all__ += vectors.base.__all__
__all__ += vectors.d1.__all__
__all__ += vectors.d2.__all__
__all__ += vectors.d3.__all__
__all__ += vectors.d4.__all__
__all__ += vectors.dn.__all__
__all__ += vectors.space.__all__
__all__ += vectors.exceptions.__all__


del vectors, install_import_hook, RUNTIME_TYPECHECKER, compat, funcs, transform
