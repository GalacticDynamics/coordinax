"""Carteisan vector."""

__all__ = [
    # Position
    "Cartesian1DVector",
    "RadialVector",
    # Differential
    "CartesianDifferential1D",
    "RadialDifferential",
]

from functools import partial
from typing import ClassVar, final

import array_api_jax_compat as xp
import equinox as eqx
import jax

from vector._checks import check_r_non_negative
from vector._typing import BatchableLength, BatchableSpeed
from vector._utils import converter_quantity_array

from .base import Abstract1DVector, Abstract1DVectorDifferential

##############################################################################
# Position


@final
class Cartesian1DVector(Abstract1DVector):
    """Cartesian vector representation."""

    x: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector."""
        return xp.abs(self.x)


@final
class RadialVector(Abstract1DVector):
    """Radial vector representation."""

    r: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)


##############################################################################
# Velocity


@final
class CartesianDifferential1D(Abstract1DVectorDifferential):
    """Cartesian differential representation."""

    d_x: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""X differential :math:`dx/dt \in (-\infty,+\infty`)`."""

    vector_cls: ClassVar[type[Cartesian1DVector]] = Cartesian1DVector  # type: ignore[misc]


@final
class RadialDifferential(Abstract1DVectorDifferential):
    """Radial differential representation."""

    d_r: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""Radial speed :math:`dr/dt \in (-\infty,+\infty)`."""

    vector_cls: ClassVar[type[RadialVector]] = RadialVector  # type: ignore[misc]
