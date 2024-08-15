"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

import os

RUNTIME_TYPECHECKER = (
    "beartype.beartype"
    if (os.environ.get("COORDINAX_ENABLE_RUNTIME_TYPECHECKS", "1") == "0")
    else None
)
