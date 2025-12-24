"""Cartesian product charts with namespaced component keys."""

__all__: tuple[str, ...] = ("CartesianProductChart",)

from dataclasses import dataclass

from typing import Any, final
from typing_extensions import override

import plum
import wadler_lindig as wl

from .base import AbstractCartesianProductChart, AbstractChart
from coordinax._src import api
from coordinax._src.custom_types import Ds, Ks


@final
@dataclass(frozen=True, slots=True)
class CartesianProductChart(AbstractCartesianProductChart[Ks, Ds]):
    """Concrete Cartesian product chart with namespaced component keys.

    Constructs a product chart from a tuple of factor charts and factor names.
    Components are namespaced tuple keys `(factor_name, component_name)` to
    avoid collisions (e.g., phase space with repeated Cart3D factors).

    Parameters
    ----------
    factors : tuple[AbstractChart, ...]
        Ordered tuple of factor charts.
    factor_names : tuple[str, ...]
        Names for each factor. Must have same length as factors and be unique.

    Examples
    --------
    >>> from coordinax._src.charts.base import CartesianProductChart
    >>> from coordinax._src.charts.euclidean import cart3d
    >>> product = CartesianProductChart((cart3d, cart3d), ("q", "p"))
    >>> product.components
    (('q', 'x'), ('q', 'y'), ('q', 'z'), ('p', 'x'), ('p', 'y'), ('p', 'z'))
    >>> product.ndim
    6

    """

    factors: tuple[AbstractChart[Any, Any], ...]
    factor_names: tuple[str, ...]

    def __post_init__(self) -> None:
        # Validate lengths match
        if len(self.factors) != len(self.factor_names):
            msg = (
                f"factors and factor_names must have the same length, "
                f"got {len(self.factors)} factors and {len(self.factor_names)} names"
            )
            raise ValueError(msg)
        # Validate unique names
        if len(set(self.factor_names)) != len(self.factor_names):
            msg = f"factor_names must be unique, got {self.factor_names}"
            raise ValueError(msg)

    # ===============================================================
    # Wadler-Lindig API

    @override
    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Wadler-Lindig pretty-printing documentation.

        TODO: write Examples

        """
        kw["include_params"] = False
        # TODO: figure out why super().__pdoc__ isn't working
        return AbstractCartesianProductChart.__pdoc__(self, **kw)


# =========================================================
# Cartesian chart for product charts


@plum.dispatch
def cartesian_chart(obj: CartesianProductChart) -> CartesianProductChart:  # type: ignore[type-arg]
    """Get Cartesian version of a namespaced product chart (factorwise).

    Returns a CartesianProductChart with each factor replaced by its
    cartesian_chart version, preserving factor_names.

    Examples
    --------
    >>> from coordinax._src.charts.base import CartesianProductChart, cartesian_chart
    >>> from coordinax._src.charts.euclidean import sph3d
    >>> product = CartesianProductChart((sph3d, sph3d), ("q", "p"))
    >>> cart_product = cartesian_chart(product)
    >>> cart_product

    """
    cart_factors = tuple(api.cartesian_chart(f) for f in obj.factors)
    # Check if already cartesian
    if cart_factors == obj.factors:
        return obj
    return CartesianProductChart(cart_factors, obj.factor_names)
