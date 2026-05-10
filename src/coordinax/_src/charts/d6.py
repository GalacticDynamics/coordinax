"""6-Dimensional charts."""

__all__ = ("Abstract6D", "PoincarePolar6D", "poincarepolar6d")


from typing import Any, Final, Literal as L, NoReturn, final  # noqa: N817
from typing_extensions import override

import jax.tree_util as jtu

from coordinax._src.base_charts import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
    is_not_abstract_chart_subclass,
)
from coordinax._src.custom_types import Len, Spd
from coordinax._src.exceptions import NoGlobalCartesianChartError


class Abstract6D(AbstractDimensionalFlag, n=6):
    """Marker flag for 6-D representations.

    An 6-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    # TODO: add a check it's 6D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)
        # n is already fixed to 6
        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


PoincarePolarKeys = tuple[
    L["rho"], L["pp_phi"], L["z"], L["dt_rho"], L["dt_pp_phi"], L["dt_z"]
]

PoincarePolarDims = tuple[
    Len, L["length / time**0.5"], Len, Spd, L["length / time**1.5"], Spd
]


@jtu.register_static
@final
@chart_dataclass_decorator
class PoincarePolar6D(
    AbstractFixedComponentsChart[PoincarePolarKeys, PoincarePolarDims], Abstract6D
):
    r"""Six-dimensional Poincare-polar chart.

    Components are ordered as
    $(\rho,\;\mathrm{pp\_phi},\;z,\;\dot\rho,\;\dot{\mathrm{pp\_phi}},\;\dot z)$
    with dimensions
    $(\mathrm{length},\;\mathrm{length}/\mathrm{time}^{1/2},\;\mathrm{length},\;\mathrm{speed},\;\mathrm{length}/\mathrm{time}^{3/2},\;\mathrm{speed})$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.poincarepolar6d.components
    ('rho', 'pp_phi', 'z', 'dt_rho', 'dt_pp_phi', 'dt_z')

    >>> cxc.poincarepolar6d.coord_dimensions
    ('length', 'length / time**0.5', 'length', 'speed', 'length / time**1.5', 'speed')

    >>> isinstance(cxc.poincarepolar6d, cxc.PoincarePolar6D)
    True

    """

    @override
    @property
    def cartesian(self) -> NoReturn:
        """PoincarePolar6D has no global Cartesian 6D representation."""
        raise NoGlobalCartesianChartError(
            "PoincarePolar6D has no global Cartesian 6D chart."
        )


poincarepolar6d: Final = PoincarePolar6D()
"""Six-dimensional Poincare-polar chart."""
