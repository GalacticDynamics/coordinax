"""Poincare."""

__all__ = ["PoincarePolarVector"]

from typing import final

import equinox as eqx
from jaxtyping import Shaped

import unxt as u

import coordinax._src.typing as ct
from coordinax._src.distances import BatchableLength
from coordinax._src.vectors.base_pos import AbstractPos


@final
class PoincarePolarVector(AbstractPos):  # TODO: better name
    """Poincare vector + differential."""

    rho: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    pp_phi: Shaped[u.Quantity, "*#batch"] = eqx.field()  # TODO: dimension annotation
    r"""Poincare phi-like variable."""

    z: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    dt_rho: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    dt_pp_phi: Shaped[u.Quantity, "*#batch"] = eqx.field()  # TODO: dimension annotation
    r"""Poincare phi-like velocity variable."""

    dt_z: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.PoincarePolarVector._dimensionality()
        6

        """
        return 6
