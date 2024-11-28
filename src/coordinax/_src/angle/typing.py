"""Type hints for `coordinax.angle`."""

__all__: list[str] = []


from jaxtyping import Shaped

import unxt as u

from .base import AbstractAngle

#: Batchable angular-type Quantity.
BatchableAngularQuantity = Shaped[u.Quantity["angle"], "*#batch"]

#: Batchable Angle.
BatchableAngle = Shaped[AbstractAngle, "*#batch"]

#: Batchable Angle or Angular Quantity.
BatchableAngleQ = BatchableAngle | BatchableAngularQuantity
