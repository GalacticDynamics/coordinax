"""Charts."""

__all__ = (
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "CHART_CLASSES",
    "NON_ABC_CHART_CLASSES",
    "chart_dataclass_decorator",
    "is_not_abstract_chart_subclass",
    "MISSING",
    "CDictT",
)

import abc
import dataclasses
import inspect
import weakref

from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal as L,  # noqa: N817
    TypeVar,
    cast,
    get_args,
    no_type_check,
)

import jax.tree_util as jtu
import plum
import wadler_lindig as wl

import dataclassish
import unxt as u

from .manifold import AbstractManifold
from coordinax._src.custom_types import CDictT, Ds, Ks

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # ty: ignore[invalid-type-form]
V = TypeVar("V")

# Charts are registered in CHART_CLASSES when they are defined, via
# AbstractChart.__init_subclass__. This allows us to find all chart classes for
# dispatch and other purposes. We use a weak set to avoid keeping classes alive
# unnecessarily, and a mapping proxy to prevent modification of the set from
# outside this module.
CHART_CLASSES: weakref.WeakSet[type["AbstractChart[Any, Any]"]] = weakref.WeakSet()

NON_ABC_CHART_CLASSES: weakref.WeakSet[type["AbstractChart[Any, Any]"]] = (
    weakref.WeakSet()
)

chart_dataclass_decorator = dataclasses.dataclass(
    frozen=True, slots=False, repr=False, eq=False
)

MISSING = object()


class MissingDefault:
    """Sentinel for missing default values in dataclass fields."""

    @property
    def default(self) -> object:
        return MISSING


MISSINGDEFAULT = MissingDefault()


##############################################################################
# AbstractChart


@jtu.register_static
@chart_dataclass_decorator
class AbstractChart(Generic[Ks, Ds], metaclass=abc.ABCMeta):
    """Abstract base class for charts (coordinate representations)."""

    _: dataclasses.KW_ONLY

    M: AbstractManifold
    """The manifold that this chart belongs to.

    Default is `no_manifold` for charts that do not belong to any manifold.
    """

    def __init_subclass__(cls, **kw: Any) -> None:
        # This allows multiple inheritance with other ABCs that might or might
        # not define an `__init_subclass__`
        if hasattr(cls, "__init_subclass__"):
            super().__init_subclass__(**kw)

        # Register the representation/chart
        # dataclass(slots=True) triggers __init_subclass__ twice:
        # 1st: class has __dataclass_params__ but no __slots__ in __dict__
        # 2nd: rebuilt class has both — this is the real one
        if (
            "__dataclass_params__" in cls.__dict__
            and "__slots__" not in cls.__dict__
            and cls.__dict__["__dataclass_params__"].slots
        ):
            return
        CHART_CLASSES.add(cls)
        if not is_abstract_class(cls):
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
    @abc.abstractmethod
    def cartesian(self) -> "AbstractChart[Ks, Ds]":
        """Return the corresponding Cartesian chart."""
        raise NotImplementedError  # pragma: no cover

    def check_data(
        self, data: CDictT, /, *, keys: bool = True, values: bool = False
    ) -> CDictT:
        """Check that the data is compatible with the chart.

        Parameters
        ----------
        data
            The data to check.
        keys
            Whether to check that the keys of `data` match `chart.components`.
            If `False`, this check is skipped.
            Default is `True`.
        values
            Whether to check that the dimensions of the values in `data` match
            `chart.coord_dimensions`. If `False`, this check is skipped.
            Default is `False`.

        """
        # Check that the keys of data match chart.components
        if keys and set(data.keys()) != set(self.components):
            msg = (
                "Data keys do not match chart components: "
                f"{set(data.keys())} != {set(self.components)}"
            )
            raise ValueError(msg)

        # Check that the dimensions of the values in data match chart.coord_dimensions
        if values:
            for k, dim in zip(self.components, self.coord_dimensions, strict=True):
                v = data[k]
                if dim is not None and u.dimension_of(v) != dim:
                    msg = (
                        f"Data dimension for '{k}' does not match chart coordinate "
                        f"dimension: {u.dimension_of(v)} != {dim}"
                    )
                    raise ValueError(msg)

        return data

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
        Cart3D[('x', 'y', 'z'), ('length', 'length', 'length')](M=Rn(3))

        >>> wl.pprint(cxc.sph3d)
        Spherical3D[('r', 'theta', 'phi'), ('length', 'angle', 'angle')](M=Rn(3))

        >>> wl.pprint(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(20, "km")))
        ProlateSpheroidal3D[('mu', 'nu', 'phi'), ('area', 'area', 'angle')](
            M=Rn(3), Delta=StaticQuantity(i64[](numpy), unit='km')
        )

        >>> wl.pprint(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(20, "km")),
        ... short_arrays=False)
        ProlateSpheroidal3D[('mu', 'nu', 'phi'), ('area', 'area', 'angle')](
            M=Rn(3), Delta=StaticQuantity(array(20), unit='km')
        )

        """
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_units", False)
        kw.setdefault("hide_defaults", True)

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

        defaults = getattr(self, "__dataclass_fields__", {})
        field_items = cast("list[tuple[str, Any]]", dataclassish.field_items(self))
        docs = [
            wl.TextDoc(k)
            + wl.TextDoc("=")
            + wl.pdoc(
                v,
                include_params=(
                    include_params if not isinstance(v, u.AbstractQuantity) else False
                ),
                **kw,
            )
            for k, v in field_items
            if k == "M"
            or not kw["hide_defaults"]
            or v is not defaults.get(k, MISSINGDEFAULT).default
        ]
        return wl.bracketed(
            begin=cls_name, docs=docs, sep=wl.comma, end=wl.TextDoc(")"), indent=4
        )

    # ===============================================================
    # Plum API

    __faithful__: ClassVar[bool] = True  # for plum caching

    # ===============================================================
    # Python API

    def __repr__(self) -> str:
        return wl.pformat(self, include_params=False, hide_defaults=True, width=80)

    def __str__(self) -> str:
        return wl.pformat(self, include_params=True, hide_defaults=False, width=80)

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
        # Make sure the other object is the same type of chart
        if type(self) is not type(other):
            return NotImplemented
        # Check the components, coord_dimensions, and field values for equality
        assert isinstance(other, AbstractChart)  # noqa: S101  # for mypy
        return (
            self.components == other.components
            and self.coord_dimensions == other.coord_dimensions
            and (dataclassish.field_values(self) == dataclassish.field_values(other))
        )

    def __hash__(self) -> int:
        """Hash a chart based on its type and field values.

        Examples
        --------
        >>> import coordinax.charts as cxc

        >>> hash(cxc.Cart3D()) == hash(cxc.cart3d)
        True

        """
        return hash(
            (
                type(self),
                self.components,
                self.coord_dimensions,
                dataclassish.field_values(self),
            )
        )


@plum.dispatch
def cartesian_chart(chart: AbstractChart, /) -> AbstractChart:
    """Return the canonical Cartesian chart for a 0D chart.

    >>> import coordinax.charts as cxc
    >>> cxc.cartesian_chart(cxc.cart0d) is cxc.cart0d
    True

    """
    return chart.cartesian


def is_abstract_class(cls: type, /) -> bool:
    """Determine if a class is abstract."""
    return inspect.isabstract(cls) or cls.__name__.startswith("Abstract")


def is_not_abstract_chart_subclass(cls: type[Any], /) -> bool:
    """Check if cls is a non-abstract non-subclass of AbstractChart."""
    return not is_abstract_class(cls) and not issubclass(cls, AbstractChart)


##############################################################################
# AbstractFixedComponentsChart


@no_type_check
def _get_tuple(tp: GAT, /) -> GAT:
    return tuple(arg.__args__[0] for arg in get_args(tp))


class AbstractFixedComponentsChart(AbstractChart[Ks, Ds]):
    """Abstract base class for charts with fixed components and dimensions."""

    _components: Ks
    _coord_dimensions: Ds

    def __init_subclass__(cls, **kw: Any) -> None:
        # Extract Ks and Ds from AbstractFixedComponentsChart in the inheritance
        if not is_abstract_class(cls):
            for base in getattr(cls, "__orig_bases__", ()):
                origin = getattr(base, "__origin__", None)
                if inspect.isclass(origin) and issubclass(
                    origin, AbstractFixedComponentsChart
                ):
                    args = get_args(base)
                    if len(args) != 2:
                        raise TypeError
                    cls._components = _get_tuple(args[0])
                    cls._coord_dimensions = _get_tuple(args[1])
                    break

        super().__init_subclass__(**kw)  # AbstractChart has.

    @property
    def components(self) -> Ks:
        return self._components

    @property
    def coord_dimensions(self) -> Ds:
        return self._coord_dimensions


##############################################################################


@jtu.register_static
class AbstractDimensionalFlag:
    """Marker base class for dimension *flags*.

    A dimension flag is a lightweight mixin used for typing and dispatch. Flags
    do not store data; instead, they classify a chart. These flags must be
    combined with concrete subclasses of {class}`AbstractChart`.


    """

    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        if n is not None:
            DIMENSIONAL_FLAGS[n] = cls

        # Enforce that this is a subclass of AbstractChart unless it's an
        # abstract base class (name starts with "Abstract")
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        # Call super() if it defines __init_subclass__
        if callable(super_init_subclass := getattr(super(), "__init_subclass__", None)):
            super_init_subclass(**kw)


DIMENSIONAL_FLAGS: Final[dict[int | L["N"], type[AbstractDimensionalFlag]]] = {}
