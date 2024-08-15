"""Poincare."""

__all__ = ["PoincarePolarVector"]

from functools import partial
from typing import final

import equinox as eqx
from jaxtyping import Shaped

from unxt import Quantity

import coordinax._coordinax.typing as ct
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.base_vel import AbstractVelocity
from coordinax._coordinax.utils import classproperty


@final
class PoincarePolarVector(AbstractPosition):  # TODO: better name
    """Poincare vector + differential."""

    rho: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    pp_phi: Shaped[Quantity["m / s(1/2)"], "*#batch"] = eqx.field()
    r"""Poincare phi-like variable."""

    z: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    d_rho: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    d_pp_phi: Shaped[Quantity["m / s(1/2)"], "*#batch"] = eqx.field()
    r"""Poincare phi-like velocity variable."""

    d_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractPosition]:
        """Return the corresponding Cartesian vector class."""
        raise NotImplementedError

    @classproperty
    @classmethod
    def differential_cls(cls) -> type[AbstractVelocity]:
        """Return the corresponding differential vector class."""
        raise NotImplementedError
