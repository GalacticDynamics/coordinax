"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__: list[str] = []

from typing import Any, TypeAlias

import unxt as u

ParamsDict: TypeAlias = dict[str, Any]
AuxDict: TypeAlias = dict[str, Any]
OptAuxDict: TypeAlias = AuxDict | None
OptUSys: TypeAlias = u.AbstractUnitSystem | None
