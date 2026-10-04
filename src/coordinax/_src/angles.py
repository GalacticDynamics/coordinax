"""Type hints for angles."""

__all__ = ("BatchableAngle", "BatchableAngleQ")


from jaxtyping import Shaped

import unxt as u

from coordinax._src.custom_types import DimQuantity

#: Batchable angular-type Quantity.
BatchableAngularQuantity = Shaped[DimQuantity["angle"], "*#batch"]

#: Batchable Angle.
BatchableAngle = Shaped[u.quantity.AbstractAngle, "*#batch"]

#: Batchable Angle or Angular Quantity.
BatchableAngleQ = BatchableAngle | BatchableAngularQuantity
