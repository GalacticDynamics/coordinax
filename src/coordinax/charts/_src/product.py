"""Cartesian product charts with namespaced component keys."""

__all__: tuple[str, ...] = (
    "AbstractCartesianProductChart",
    "AbstractFlatCartesianProductChart",
    "CartesianProductChart",
)

import abc
from itertools import chain

from collections.abc import Mapping
from typing import Any, TypeVar, final

import plum
import wadler_lindig as wl  # type: ignore[import-untyped]  # type: ignore[import-untyped]

import coordinax.api.charts as api
from .base import MISSING, AbstractChart, chart_dataclass_decorator
from .custom_types import CDict
from coordinax.internal.custom_types import Ds, Ks, OptUSys

V = TypeVar("V")


class AbstractCartesianProductChart(AbstractChart[Ks, Ds]):
    """Abstract base class for Cartesian product charts.

    A Cartesian product chart is defined by a finite ordered tuple of factor
    charts. The product chart's components are the concatenation of factor
    components, and transformations follow factorwise laws.

    Mathematical definition:
        M = ∏ᵢ Mᵢ  (product manifold)
        TₚM ≅ ⊕ᵢ T_{pᵢ}Mᵢ  (tangent bundle splits)

    Component key convention (normative):
        - General product charts use **dot-delimited** string keys:
          ``"factor_name.c"`` where ``factor_name`` identifies the factor and
          ``c`` is the factor's component key.
        - Specialized products (e.g. `SpaceTimeCT`) may
          expose **flat** string keys as a documented exception when component
          names are guaranteed collision-free (e.g. "ct", "x", "y", "z").

    Normative requirements:
    - `factors` is an ordered tuple of charts
    - `factor_names` is always a tuple of unique strings with length matching `factors`
    - `ndim` must equal `sum(f.ndim for f in factors)`
    - `components` must follow the key convention above
    """

    @property
    @abc.abstractmethod
    def factors(self) -> tuple["AbstractChart[Any, Any]", ...]:
        """Ordered tuple of factor charts."""

    @property
    @abc.abstractmethod
    def factor_names(self) -> tuple[str, ...]:
        """Factor names for namespaced keys.

        Must return a tuple of unique strings aligned with `factors`.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def ndim(self) -> int:
        """Total dimension: sum of factor dimensions."""
        return sum(f.ndim for f in self.factors)

    @property
    def components(self) -> Ks:
        """Component keys: dot-delimited strings ``"factor_name.component"``.

        Components are dot-delimited string keys to avoid collisions:
            (f"{name_0}.{c}" for c in factors[0].components, ...)
        """
        return tuple(  # Namespaced dot-delimited keys via a single comprehension
            f"{name}.{c}"
            for name, factor in zip(self.factor_names, self.factors, strict=True)
            for c in factor.components
        )  # type: ignore[return-value]

    @property
    def coord_dimensions(self) -> Ds:
        """Concatenation of factor coordinate dimensions."""
        return tuple(d for f in self.factors for d in f.coord_dimensions)  # type: ignore[return-value]

    def split_components(self, p: CDict, /) -> tuple[CDict, ...]:
        """Partition a CDict by factor components.

        For namespaced products: select keys ``"name_i.c"`` and strip prefix
        to yield factor dict keyed by ``c``.

        Parameters
        ----------
        p :
            Point dictionary with keys matching this chart's components.

        Returns
        -------
        tuple[CDict, ...]
            Tuple of dictionaries, one per factor, with factor-native keys.

        """
        p_get = p.get
        # Namespaced: extract by prefix and strip using a single pass per factor
        return tuple(
            {
                c: v
                for c in factor.components
                if (v := p_get(f"{name}.{c}", MISSING)) is not MISSING
            }
            for name, factor in zip(self.factor_names, self.factors, strict=True)
        )

    def merge_components(self, parts: tuple[Mapping[str, V], ...], /) -> dict[str, V]:
        """Merge factor CDicts into a single CDict.

        For namespaced products: re-attach dot-delimited prefix
        ``"name_i.c"`` for each factor key.

        Parameters
        ----------
        parts : tuple[Mapping[str, V], ...]
            Tuple of dictionaries, one per factor, with factor-native keys.

        Returns
        -------
        dict[str, V]
            Merged dictionary in this chart's component order.

        """
        # Namespaced: add dot-delimited prefix
        return {
            f"{name}.{c}": v
            for name, part in zip(self.factor_names, parts, strict=True)
            for c, v in part.items()
        }

    def __pdoc__(self, *, include_params: bool = False, **kw: Any) -> wl.AbstractDoc:
        return super().__pdoc__(include_params=include_params, **kw)


MSG_COMPONENT_KEY_COLLISION = (
    "Component key collision in flat-key product chart "
    "{chart.__class__.__name__}. Factors have overlapping component "
    "names: {comps} + {c}. Use factor_names for namespacing."
)


class AbstractFlatCartesianProductChart(AbstractCartesianProductChart[Ks, Ds]):
    """Abstract base class for flat-key Cartesian product charts.

    A flat-key Cartesian product chart is a specialization of
    :class:`AbstractCartesianProductChart` where the component keys are
    guaranteed to be collision-free across factors, allowing the use of
    flat string keys instead of dot-delimited string keys.

    Subclasses must provide factor_names and ensure that factor components do not
    collide.

    Normative requirements:
    - `factor_names` must be provided (abstract property)
    - `components` is the direct concatenation of factor components
      (must be collision-free)
    """

    @property
    @abc.abstractmethod
    def factor_names(self) -> tuple[str, ...]:
        """Factor names for flat-key Cartesian product charts.

        Subclasses must provide factor names even though components are flat.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def components(self) -> Ks:
        """Component keys: flat strings (collision-free concatenation of factors).

        Components are the direct concatenation of factor component strings
        (must be collision-free).
        """
        # Flat keys (specialized products like SpaceTimeCT)
        seen: set[str] = set()
        flat_components: list[str] = []
        seen_add = seen.add
        fc_append = flat_components.append
        for c in chain.from_iterable(f.components for f in self.factors):
            if c in seen:
                msg = MSG_COMPONENT_KEY_COLLISION.format(
                    chart=self, comps=flat_components, c=c
                )
                raise ValueError(msg)
            seen_add(c)
            fc_append(c)
        return tuple(flat_components)  # type: ignore[return-value]

    def split_components(self, p: CDict, /) -> tuple[CDict, ...]:
        """Partition a CDict by factor components.

        For flat-key products: partition by ``factor.components`` directly.

        Parameters
        ----------
        p :
            Point dictionary with keys matching this chart's components.

        Returns
        -------
        tuple[CDict, ...]
            Tuple of dictionaries, one per factor, with factor-native keys.

        """
        # partition by factor.components with cached lookup
        p_get = p.get
        return tuple(
            {c: v for c in factor.components if (v := p_get(c, MISSING)) is not MISSING}
            for factor in self.factors
        )

    def merge_components(self, parts: tuple[Mapping[str, V], ...]) -> dict[str, V]:
        """Merge factor CDicts into a single CDict.

        For flat-key products: merge as-is (keys already match components).

        Parameters
        ----------
        parts : tuple[Mapping[str, V], ...]
            Tuple of dictionaries, one per factor, with factor-native keys.

        Returns
        -------
        dict[str, V]
            Merged dictionary in this chart's component order.

        """
        return {k: v for part in parts for k, v in part.items()}


# =========================================================


@final
@chart_dataclass_decorator
class CartesianProductChart(AbstractCartesianProductChart[Ks, Ds]):
    """Concrete Cartesian product chart with dot-delimited component keys.

    Constructs a product chart from a tuple of factor charts and factor names.
    Components are dot-delimited string keys ``"factor_name.component_name"``
    to avoid collisions (e.g., phase space with repeated Cart3D factors).

    Parameters
    ----------
    factors : tuple[AbstractChart, ...]
        Ordered tuple of factor charts.
    factor_names : tuple[str, ...]
        Names for each factor. Must have same length as factors and be unique.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> product = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
    >>> product.components
    ('q.x', 'q.y', 'q.z', 'p.x', 'p.y', 'p.z')
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


# =========================================================
# Cartesian chart for product charts


@plum.dispatch
def cartesian_chart(obj: CartesianProductChart) -> CartesianProductChart:  # type: ignore[type-arg]
    """Get Cartesian version of a namespaced product chart (factorwise).

    Returns a CartesianProductChart with each factor replaced by its
    cartesian_chart version, preserving factor_names.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> product = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
    >>> cart_product = cxc.cartesian_chart(product)
    >>> cart_product
    CartesianProductChart(factors=(Cart3D(), Cart3D()), factor_names=('q', 'p'))

    """
    cart_factors = tuple(api.cartesian_chart(f) for f in obj.factors)
    # Check if already cartesian
    if cart_factors == obj.factors:
        return obj
    return CartesianProductChart(cart_factors, obj.factor_names)


# ===================================================================
# Cartesian Product Chart conversions


@plum.dispatch
def point_transition_map(
    to_chart: AbstractCartesianProductChart,  # type: ignore[type-arg]
    from_chart: AbstractCartesianProductChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    r"""AbstractCartesianProductChart -> AbstractCartesianProductChart (factorwise).

    Transforms between product charts by applying
    {func}`~coordinax.charts.point_realization_map` to each factor
    independently. Requires compatible factor structure (same number of factors,
    pairwise compatible).

    Mathematical definition:

    $$ \varphi \left(\prod_i S_i,\;\prod_i R_i,\;p\right)
        = \bigl(\varphi(S_i,\,R_i,\,p_i)\bigr)_i $$

    where $\varphi$ denotes {func}`~coordinax.charts.point_realization_map` and
    $p_i$ are the factor dictionaries split from $p$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Transform SpaceTimeCT (a product chart) between spatial representations:

    >>> spacetime_cart = cxc.SpaceTimeCT(cxc.cart3d)
    >>> spacetime_sph = cxc.SpaceTimeCT(cxc.sph3d)
    >>> p = {"ct": u.Q(1.0, "s"), "x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"),
    ...      "z": u.Q(0.0, "m")}
    >>> result = cxc.point_transition_map(spacetime_sph, spacetime_cart, p)
    >>> result["ct"]
    Q(1., 's')
    >>> result["r"]
    Q(1., 'm')

    """
    # Product charts can't safely use the cartesian intermediate because their
    # cartesian version is still a product chart, which would recurse here. Do
    # a factor-wise transform directly instead.
    if len(to_chart.factors) != len(from_chart.factors):
        msg = (
            "Cannot transform between product charts with different numbers of "
            "factors: "
            f"{len(from_chart.factors)} -> {len(to_chart.factors)}"
        )
        raise TypeError(msg)

    parts = from_chart.split_components(p)
    transformed = tuple(
        api.point_transition_map(t_factor, f_factor, p_part, usys=usys)
        for t_factor, f_factor, p_part in zip(
            to_chart.factors, from_chart.factors, parts, strict=True
        )
    )
    return to_chart.merge_components(transformed)


@plum.dispatch
def point_transition_map(
    to_chart: AbstractCartesianProductChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> Cartesian -> AbstractChart."""
    del p, usys
    msg = (
        f"No general transform between {type(from_chart).__name__} and "
        f"{type(to_chart).__name__}. Define explicit rules for non-product to "
        "product conversions."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractCartesianProductChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> Cartesian -> AbstractChart."""
    del p, usys
    msg = (
        f"No general transform between {type(from_chart).__name__} and "
        f"{type(to_chart).__name__}. Define explicit rules for product to "
        "non-product conversions."
    )
    raise NotImplementedError(msg)
