"""6-Dimensional charts."""

__all__ = ("Abstract6D", "PoincarePolar6D", "poincarepolar6d")


import dataclasses

from typing import Any, Final, Literal as L, NoReturn, override  # noqa: N817

from coordinax._src.base import (
    AbstractDimensionalFlag,
    AbstractManifold,
    AbstractStaticFixedComponentsChart,
    chart_dataclass_decorator,
)
from coordinax._src.custom_types import Len, Spd
from coordinax._src.exceptions import NoGlobalCartesianChartError
from coordinax._src.null import no_manifold


class Abstract6D(AbstractDimensionalFlag, n=6):
    """Marker flag for 6-D representations.

    An 6-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # n is already fixed to 6
        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


PoincarePolarKeys = tuple[
    L["rho"], L["pp_phi"], L["z"], L["dt_rho"], L["pp_phidot"], L["dt_z"]
]

PoincarePolarDims = tuple[
    Len, L["length / time**0.5"], Len, Spd, L["length / time**0.5"], Spd
]


@chart_dataclass_decorator
class PoincarePolar6D(
    AbstractStaticFixedComponentsChart[Any, PoincarePolarKeys, PoincarePolarDims],
    Abstract6D,
):
    r"""Six-dimensional Poincaré symplectic-polar chart on phase space.

    A chart on the 6-D phase space $T^*\mathbb{R}^3$ in the *Poincaré symplectic
    polar* variables used in galactic dynamics. The azimuthal action-angle pair
    $(\phi, L_z)$ is replaced by the Cartesian-like quadrature pair

    $\mathrm{pp\_phi} = \sqrt{2\,|L_z|}\,\cos\phi,$
    $\mathrm{pp\_phidot} = \sqrt{2\,|L_z|}\,\sin\phi,$

    which removes the coordinate singularity of polar coordinates on the axis.
    Following {mod}`gala` (``cartesian_to_poincare_polar``; Papaphilippou &
    Laskar 1996, A&A 307, 427) the components are ordered as
    $(\rho,\;\mathrm{pp\_phi},\;z,\;\dot\rho,\;\mathrm{pp\_phidot},\;\dot z)$
    with dimensions
    $(\mathrm{length},\;\mathrm{length}/\mathrm{time}^{1/2},\;\mathrm{length},\;\mathrm{speed},\;\mathrm{length}/\mathrm{time}^{1/2},\;\mathrm{speed})$.

    ``pp_phi`` and ``pp_phidot`` are the two symplectic quadratures of the
    azimuth (both $\sqrt{\text{action}}$, hence the ``length / time**0.5``
    dimension); ``pp_phidot`` is **not** the time derivative of ``pp_phi``
    despite the ``dot`` suffix (it is gala's ``p_phi_dot``). ``rho``, ``z`` are
    cylindrical position and ``dt_rho``, ``dt_z`` their velocities.

    Notes
    -----
    Phase space carries a *symplectic* form $\omega$, not a Riemannian metric,
    so this chart has **no manifold metric**: ``M`` is ``no_manifold`` and there
    is no ``metric_matrix``. It likewise has no global 6-D Cartesian chart
    (``cartesian`` raises). The forward map discards $\mathrm{sign}(L_z)$ (via
    $\sqrt{|L_z|}$), so it is not injective and the inverse is ambiguous in the
    sign of angular momentum — matching gala, which provides only the forward
    transform.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.poincarepolar6d.components
    ('rho', 'pp_phi', 'z', 'dt_rho', 'pp_phidot', 'dt_z')

    >>> cxc.poincarepolar6d.coord_dimensions
    ('length', 'length / time**0.5', 'length', 'speed', 'length / time**0.5', 'speed')

    >>> isinstance(cxc.poincarepolar6d, cxc.PoincarePolar6D)
    True

    """

    _: dataclasses.KW_ONLY
    # Phase space is symplectic, not Riemannian: there is no manifold metric,
    # so `M` is `no_manifold` (a metric / `metric_matrix` is inapplicable).
    M: AbstractManifold = no_manifold

    @override
    @property
    def cartesian(self) -> NoReturn:
        """PoincarePolar6D has no global Cartesian 6D representation."""
        raise NoGlobalCartesianChartError(
            "PoincarePolar6D has no global Cartesian 6D chart."
        )


poincarepolar6d: Final = PoincarePolar6D()
"""Six-dimensional Poincaré symplectic-polar phase-space chart."""
