"""Built-in vector classes."""

__all__ = ["ProlateSpheroidalAcc", "ProlateSpheroidalPos", "ProlateSpheroidalVel"]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
from jaxtyping import Shaped

import quaxed.numpy as jnp
from dataclassish.converters import Unless
from unxt import Quantity

import coordinax._src.typing as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.angle import Angle
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import VectorAttribute
from coordinax._src.vectors.checks import (
    check_greater_than_equal,
    check_less_than_equal,
    check_non_negative_non_zero,
)
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class ProlateSpheroidalPos(AbstractPos3D):
    """Prolate spheroidal coordinates as defined by Dejonghe & de Zeeuw 1988.

    Note that valid coordinates have:
    - mu >= Delta^2
    - |nu| <= Delta^2
    - Delta > 0

    Parameters
    ----------
    mu : Quantity["area"]
        The spheroidal mu coordinate. This is called `lambda` by Dejonghe & de Zeeuw.
    nu : Quantity["area"]
        The spheroidal nu coordinate.
    phi : `coordinax.angle.Angle`
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    Delta : Quantity["length"]
        The focal length of the coordinate system. Must be > 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(3.0, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.25, "rad"),
    ...     Delta=Quantity(1.5, "kpc"),
    ... )
    >>> vec
    ProlateSpheroidalPos(
      mu=Quantity[PhysicalType('area')](value=f32[], unit=Unit("kpc2")),
      nu=Quantity[PhysicalType('area')](value=f32[], unit=Unit("kpc2")),
      phi=Angle(value=f32[], unit=Unit("rad")),
      Delta=Quantity[PhysicalType('length')](value=weak_f32[], unit=Unit("kpc"))
    )

    This fails with a zero or negative Delta:

    >>> try: vec = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(3.0, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.25, "rad"),
    ...     Delta=Quantity(0.0, "kpc"),
    ... )
    ... except Exception as e: pass

    Or with invalid mu and nu:

    >>> try: vec = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(0.5, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.25, "rad"),
    ...     Delta=Quantity(1.5, "kpc"),
    ... )
    ... except Exception as e: pass

    """

    mu: ct.BatchableArea = eqx.field(
        converter=partial(Quantity["area"].from_, dtype=float)
    )
    r"""Spheroidal mu coordinate :math:`\mu \in [0,+\infty)` (called :math:`\lambda` in
     some Galactic dynamics contexts)."""

    nu: ct.BatchableArea = eqx.field(
        converter=partial(Quantity["area"].from_, dtype=float)
    )
    r"""Spheroidal nu coordinate :math:`\lambda \in [-\infty,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=Unless(
            Angle, lambda x: converter_azimuth_to_range(Angle.from_(x, dtype=float))
        )
    )
    r"""Azimuthal angle, generally :math:`\phi \in [0,360)`."""

    _: KW_ONLY
    Delta: Shaped[Quantity["length"], ""] = VectorAttribute(
        converter=Quantity["length"].from_
    )
    """Focal length of the coordinate system."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_non_negative_non_zero(self.Delta, name="Delta")
        check_greater_than_equal(
            self.mu, self.Delta**2, name="mu", comparison_name="Delta^2"
        )
        check_less_than_equal(
            jnp.abs(self.nu), self.Delta**2, name="nu", comparison_name="Delta^2"
        )

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["ProlateSpheroidalVel"]:
        return ProlateSpheroidalVel


@final
class ProlateSpheroidalVel(AbstractVel3D):
    """Prolate spheroidal differential representation."""

    d_mu: ct.BatchableDiffusivity = eqx.field(
        converter=partial(Quantity["diffusivity"].from_, dtype=float)
    )
    r"""Prolate spheroidal mu speed :math:`d\mu/dt \in [-\infty, \infty]."""

    d_nu: ct.BatchableDiffusivity = eqx.field(
        converter=partial(Quantity["diffusivity"].from_, dtype=float)
    )
    r"""Prolate spheroidal nu speed :math:`d\nu/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].from_, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[ProlateSpheroidalPos]:
        return ProlateSpheroidalPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["ProlateSpheroidalAcc"]:
        return ProlateSpheroidalAcc


@final
class ProlateSpheroidalAcc(AbstractAcc3D):
    """Prolate spheroidal acceleration representation."""

    d2_mu: ct.BatchableSpecificEnergy = eqx.field(
        converter=partial(Quantity["specific energy"].from_, dtype=float)
    )
    r"""Prolate spheroidal mu acceleration :math:`d^2\mu/dt^2 \in [-\infty, \infty]."""

    d2_nu: ct.BatchableSpecificEnergy = eqx.field(
        converter=partial(Quantity["specific energy"].from_, dtype=float)
    )
    r"""Prolate spheroidal nu acceleration :math:`d^2\nu/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].from_, dtype=float)
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[ProlateSpheroidalVel]:
        return ProlateSpheroidalVel
