"""Vector."""

from unxt.quantity import is_any_quantity

__all__ = (
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "CHART_CLASSES",
    "AbstractCartesianProductChart",
    "AbstractFlatCartesianProductChart",
)

import abc
import weakref
from itertools import chain

from collections.abc import Mapping
from typing import (
    Any,
    Final,
    Generic,
    Literal as L,  # noqa: N817
    TypeVar,
    get_args,
    no_type_check,
)

import plum
import wadler_lindig as wl  # type: ignore[import-untyped]

import unxt as u
from dataclassish import field_items
from unxt.quantity import is_any_quantity

from coordinax._src.custom_types import Ds, Ks
from coordinax.api import CDict, CsDict, cartesian_chart, metric_of

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # type: ignore[misc]
V = TypeVar("V")

CHART_CLASSES: weakref.WeakSet[type["AbstractChart[Any, Any]"]] = weakref.WeakSet()
NON_ABC_CHART_CLASSES: weakref.WeakSet[type["AbstractChart[Any, Any]"]] = (
    weakref.WeakSet()
)

MISSING = object()


class AbstractChart(Generic[Ks, Ds], metaclass=abc.ABCMeta):
    """Abstract base class for charts (coordinate representations)."""

    def __init_subclass__(cls, **kw: Any) -> None:
        # This allows multiple inheritance with other ABCs that might or might
        # not define an `__init_subclass__`
        if hasattr(cls, "__init_subclass__"):
            super().__init_subclass__(**kw)

        # Register the representation/chart
        CHART_CLASSES.add(cls)
        if not cls.__name__.startswith("Abstract"):
            NON_ABC_CHART_CLASSES.add(cls)

    # ===============================================================
    # Vector API

    @property
    @abc.abstractmethod
    def components(self) -> Ks:
        """The names of the components."""
        ...

    @property
    @abc.abstractmethod
    def coord_dimensions(self) -> Ds:
        """The dimensions of the components."""
        ...

    @property
    def ndim(self) -> int:
        """Number of coordinate components (chart dimension)."""
        return len(self.components)

    @property
    def cartesian(self) -> "AbstractChart[Ks, Ds]":
        """Return the corresponding Cartesian vector class."""
        return cartesian_chart(self)

    @property
    def is_euclidean(self) -> bool:
        r"""Check if this representation lives in a Euclidean (free-vector) space.

        Mathematical definition
        -----------------------
        Returns ``True`` when the representation's metric is Euclidean, meaning
        the tangent space at every point is isomorphic to the underlying
        $\mathbb{R}^n$ and displacements can be defined without a base
        point (free vectors).

        $$
           \mathrm{is\_euclidean} =
           \begin{cases}
             \mathtt{True}  & g_{\mathrm{rep}} = \delta_{ij} \\
             \mathtt{False} & \text{otherwise}
           \end{cases}
        $$

        Returns
        -------
        bool
            ``True`` if the representation's metric is ``EuclideanMetric``,
            ``False`` otherwise.

        Notes
        -----
        This property is used to determine whether vector addition requires
        an ``at=`` base-point parameter:

        - Euclidean reps: addition is defined anywhere; no ``at=`` needed.
        - Non-Euclidean reps (sphere, Minkowski, etc.): displacement addition
          requires a base point or embedding.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> cxc.cart3d.is_euclidean
        True

        >>> cxc.sph3d.is_euclidean
        True

        >>> cxc.twosphere.is_euclidean
        False

        """
        # Function-local to avoid a circular import
        from coordinax.metrics import EuclideanMetric  # noqa: PLC0415

        return isinstance(metric_of(self), EuclideanMetric)

    def check_data(self, data: CsDict, /) -> None:
        # Check that the keys of data match chart.components
        if set(data.keys()) != set(self.components):
            msg = (
                "Data keys do not match chart components: "
                f"{set(data.keys())} != {set(self.components)}"
            )
            raise ValueError(msg)

        # Allowing for data to be a Array-dict
        # Check that the dimensions match chart.coord_dimensions
        if any(map(is_any_quantity, data.values())):
            for k, d in zip(self.components, self.coord_dimensions, strict=True):
                v = data[k]
                if d is not None and u.dimension_of(v) != d:
                    msg = "Data dimensions do not match"
                    raise ValueError(msg)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, include_params: bool = True, **kw: Any) -> wl.AbstractDoc:
        """Wadler-Lindig pretty-printing documentation.

        All keyword arguments are passed to :func:`wadler_lindig.pdoc` for the
        field values. Most AbstractChart subclasses do not have any fields.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import unxt as u
        >>> import wadler_lindig as wl

        >>> wl.pprint(cxc.cart3d)
        Cart3D[('x', 'y', 'z'), ('length', 'length', 'length')]()

        >>> wl.pprint(cxc.sph3d)
        Spherical3D[('r', 'theta', 'phi'), ('length', 'angle', 'angle')]()

        >>> wl.pprint(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(20, "km")))
        ProlateSpheroidal3D[('mu', 'nu', 'phi'), ('area', 'area', 'angle')](
            Delta=StaticQuantity(i64[](numpy), unit='km')
        )

        >>> wl.pprint(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(20, "km")),
        ... short_arrays=False)
        ProlateSpheroidal3D[('mu', 'nu', 'phi'), ('area', 'area', 'angle')](
            Delta=StaticQuantity(array(20), unit='km')
        )

        """
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_units", False)

        if include_params:
            cls_name = wl.bracketed(
                begin=wl.TextDoc(f"{self.__class__.__name__}["),
                docs=[
                    wl.pdoc(self.components, **kw),
                    wl.pdoc(self.coord_dimensions, **kw),
                ],
                sep=wl.comma,
                end=wl.TextDoc("]("),
                indent=2,
            )
        else:
            cls_name = wl.TextDoc(f"{self.__class__.__name__}(")

        fields = wl.bracketed(
            begin=wl.TextDoc(""),
            docs=wl.named_objs(list(field_items(self)), **kw),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )
        return cls_name + fields

    # ===============================================================
    # Python API

    def __repr__(self) -> str:
        return wl.pformat(self, include_params=False, width=80)

    def __str__(self) -> str:
        return wl.pformat(self, include_params=True, width=80)

    def __eq__(self, other: object) -> bool:
        """Check equality between charts.

        Two charts are equal if they are the same type and have equal field
        values.

        Examples
        --------
        >>> import coordinax.charts as cxc

        >>> cxc.Cart3D() == cxc.cart3d
        True

        >>> cxc.Cart3D() == cxc.sph3d
        False

        """
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.components == other.components
            and self.coord_dimensions == other.coord_dimensions
            and (field_values(self) == field_values(other))
        )

    def __hash__(self) -> int:
        """Hash a chart based on its type and field values.

        Examples
        --------
        >>> import coordinax.charts as cxc

        >>> hash(cxc.Cart3D()) == hash(cxc.cart3d)
        True

        """
        return hash((type(self), field_values(self)))


@plum.dispatch
def field_values(
    obj: AbstractChart,  # type: ignore[type-arg]
) -> tuple[Any, ...]:
    fields = getattr(obj, "__dataclass_fields__", {})
    return tuple(getattr(obj, f.name) for f in fields.values())


@plum.dispatch
def field_items(
    obj: AbstractChart,  # type: ignore[type-arg]
) -> tuple[Any, ...]:
    fields = getattr(obj, "__dataclass_fields__", {})
    return tuple((f.name, getattr(obj, f.name)) for f in fields.values())


@no_type_check
def get_tuple(tp: GAT, /) -> GAT:
    return tuple(arg.__args__[0] for arg in get_args(tp))


class AbstractFixedComponentsChart(AbstractChart[Ks, Ds]):
    """Abstract base class for charts with fixed components and dimensions."""

    def __init_subclass__(cls, **kw: Any) -> None:
        # Extract Ks and Ds from AbstractFixedComponentsChart in the inheritance
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is AbstractFixedComponentsChart:
                args = get_args(base)
                if len(args) != 2:
                    raise TypeError
                cls._components = get_tuple(args[0])  # type: ignore[attr-defined]
                cls._coord_dimensions = get_tuple(args[1])  # type: ignore[attr-defined]
                break

        super().__init_subclass__(**kw)  # AbstractChart has.

    @property
    def components(self) -> Ks:
        return self._components  # type: ignore[attr-defined]

    @property
    def coord_dimensions(self) -> Ds:
        return self._coord_dimensions  # type: ignore[attr-defined]


class AbstractDimensionalFlag:
    """Marker base class for dimension *flags*.

    A dimension flag is a lightweight mixin used purely for typing and
    dispatch. Flags do not store data; instead, they classify a representation
    as describing a position, velocity, acceleration, or similar semantic role.

    These flags are combined with concrete subclasses of
    :class:`AbstractChart` to define the meaning of a vector.
    """

    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        if n is not None:
            DIMENSIONAL_FLAGS[n] = cls

        # Enforce that this is a subclass of AbstractChart
        if not cls.__name__.startswith("Abstract") and not issubclass(
            cls, AbstractChart
        ):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if hasattr(super(), "__init_subclass__"):
            super().__init_subclass__(**kw)


DIMENSIONAL_FLAGS: Final[dict[int | L["N"], type[AbstractDimensionalFlag]]] = {}


# =========================================================
# Cartesian Product Charts


class AbstractCartesianProductChart(AbstractChart[Ks, Ds]):
    """Abstract base class for Cartesian product charts.

    A Cartesian product chart is defined by a finite ordered tuple of factor
    charts. The product chart's components are the concatenation of factor
    components, and transformations follow factorwise laws.

    Mathematical definition:
        M = ∏ᵢ Mᵢ  (product manifold)
        TₚM ≅ ⊕ᵢ T_{pᵢ}Mᵢ  (tangent bundle splits)

    Component key convention (normative):
        - General product charts use **namespaced** tuple keys: `(factor_name, c)`
          where `factor_name` identifies the factor and `c` is the factor's
          component key.
        - Specialized products (e.g. `SpaceTimeCT`, `SpaceTimeEuclidean`) may
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
        ...

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
        """Component keys: namespaced tuples (factor_name, component_name).

        Components are namespaced tuple keys to avoid collisions:
            ((name_0, c) for c in factors[0].components, ...)
        """
        return tuple(  # Namespaced tuple keys via a single comprehension
            (name, c)
            for name, factor in zip(self.factor_names, self.factors, strict=True)
            for c in factor.components
        )  # type: ignore[return-value]

    @property
    def coord_dimensions(self) -> Ds:
        """Concatenation of factor coordinate dimensions."""
        return tuple(d for f in self.factors for d in f.coord_dimensions)  # type: ignore[return-value]

    def split_components(self, p: CsDict, /) -> tuple[CDict, ...]:
        """Partition a CsDict by factor components.

        For namespaced products: select keys `(name_i, c)` and strip prefix to
        yield factor dict keyed by `c`.

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
                if (v := p_get((name, c), MISSING)) is not MISSING
            }
            for name, factor in zip(self.factor_names, self.factors, strict=True)
        )

    def merge_components(
        self, parts: tuple[Mapping[str, V], ...], /
    ) -> dict[str | tuple[str, str], V]:
        """Merge factor CsDicts into a single CsDict.

        For namespaced products: re-attach prefix `(name_i, c)` for each factor key.

        Parameters
        ----------
        parts : tuple[Mapping[str, V], ...]
            Tuple of dictionaries, one per factor, with factor-native keys.

        Returns
        -------
        dict[str | tuple[str, str], V]
            Merged dictionary in this chart's component order.

        """
        # Namespaced: add prefix
        return {
            (name, c): v
            for name, part in zip(self.factor_names, parts, strict=True)
            for c, v in part.items()
        }

    def __pdoc__(self, *, include_params: bool = False, **kw: Any) -> wl.AbstractDoc:
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_units", False)

        if include_params:
            cls_name = wl.bracketed(
                begin=wl.TextDoc(f"{self.__class__.__name__}["),
                docs=[
                    wl.pdoc(self.components, include_params=include_params, **kw),
                    wl.pdoc(self.coord_dimensions, include_params=include_params, **kw),
                ],
                sep=wl.comma,
                end=wl.TextDoc("]("),
                indent=2,
            )
        else:
            cls_name = wl.TextDoc(f"{self.__class__.__name__}(")

        fields = wl.bracketed(
            begin=wl.TextDoc(""),
            docs=wl.named_objs(
                list(field_items(self)), include_params=include_params, **kw
            ),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )
        return cls_name + fields


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
    flat string keys instead of namespaced tuple keys.

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
        """Component keys: namespaced tuples or flat strings depending on factor_names.

        If `factor_names` is provided, components are namespaced tuple keys:
            ((name_0, c) for c in factors[0].components, ...)

        If `factor_names` is None (flat-key specialization), components are the
        direct concatenation of factor component strings (must be collision-free).
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

    def split_components(self, p: CsDict, /) -> tuple[CDict, ...]:
        """Partition a CsDict by factor components.

        For namespaced products: select keys `(name_i, c)` and strip prefix to
        yield factor dict keyed by `c`.

        For flat-key products: partition by `factor.components` directly.

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

    def merge_components(
        self, parts: tuple[Mapping[str, V], ...]
    ) -> dict[str | tuple[str, str], V]:
        """Merge factor CsDicts into a single CsDict.

        For namespaced products: re-attach prefix `(name_i, c)` for each factor key.

        For flat-key products: merge as-is (keys already match components).

        Parameters
        ----------
        parts : tuple[Mapping[str, V], ...]
            Tuple of dictionaries, one per factor, with factor-native keys.

        Returns
        -------
        dict[str | tuple[str, str], V]
            Merged dictionary in this chart's component order.

        """
        return {k: v for part in parts for k, v in part.items()}
