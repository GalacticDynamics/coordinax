"""Poincare."""

__all__ = ["PoincarePolarVector"]

from typing import final

import equinox as eqx
from jaxtyping import Shaped

import unxt as u

import coordinax._src.typing as ct
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos, AbstractVel


@final
class PoincarePolarVector(AbstractPos):  # TODO: better name
    """Poincare vector + differential."""

    rho: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    pp_phi: Shaped[u.Quantity, "*#batch"] = eqx.field()  # TODO: dimension annotation
    r"""Poincare phi-like variable."""

    z: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    d_rho: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    d_pp_phi: Shaped[u.Quantity, "*#batch"] = eqx.field()  # TODO: dimension annotation
    r"""Poincare phi-like velocity variable."""

    d_z: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
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

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractPos]:
        """Return the corresponding Cartesian vector class."""
        raise NotImplementedError

    @classproperty
    @classmethod
    def differential_cls(cls) -> type[AbstractVel]:
        """Return the corresponding differential vector class."""
        raise NotImplementedError
