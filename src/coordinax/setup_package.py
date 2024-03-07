"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

import os

RUNTIME_TYPECHECKER = (
    "beartype.beartype"
    if (os.environ.get("VECTOR_ENABLE_RUNTIME_TYPECHECKS", "1") == "1")
    else None
)
