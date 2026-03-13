"""Vector."""

__all__ = (
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "CHART_CLASSES",
    "chart_dataclass_decorator",
)

import abc
import dataclasses
import weakref

from typing import (
    Any,
    Final,
    Generic,
    Literal as L,  # noqa: N817
    TypeVar,
    get_args,
    no_type_check,
)

import jax.tree_util as jtu
import wadler_lindig as wl  # type: ignore[import-untyped]

import unxt as u
from dataclassish import field_items, field_values

import coordinax.api.charts as api
from .custom_types import CDict
from coordinax.internal.custom_types import Ds, Ks, OptUSys

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # type: ignore[misc]
V = TypeVar("V")

CHART_CLASSES: weakref.WeakSet[type["AbstractChart[Any, Any]"]] = weakref.WeakSet()
NON_ABC_CHART_CLASSES: weakref.WeakSet[type["AbstractChart[Any, Any]"]] = (
    weakref.WeakSet()
)

chart_dataclass_decorator = dataclasses.dataclass(
    frozen=True, slots=True, repr=False, eq=False
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
        """Return the corresponding Cartesian chart."""
        return api.cartesian_chart(self)

    def check_data(self, data: CDict, /, *, components: bool = True) -> None:
        """Check that the data is compatible with the chart.

        Parameters
        ----------
        data
            The data to check.
        components
            Whether to check that the keys of `data` match `chart.components`.
            If `False`, this check is skipped.
            Default is `True`.

        """
        # Check that the keys of data match chart.components
        if components and set(data.keys()) != set(self.components):
            msg = (
                "Data keys do not match chart components: "
                f"{set(data.keys())} != {set(self.components)}"
            )
            raise ValueError(msg)

    # -------------------------------------------
    # Point-role ambient Cartesian realization

    def realize_cartesian(self, data: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Realize a point in canonical ambient Cartesian coordinates."""
        return api.realize_cartesian(self, data, usys=usys)

    # TODO: maybe remove.
    def unrealize_cartesian(self, data: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Invert the ambient Cartesian realization on the chart domain."""
        return api.unrealize_cartesian(self, data, usys=usys)

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
            for k, v in field_items(self)
            if not kw["hide_defaults"]
            or v is not defaults.get(k, MISSINGDEFAULT).default
        ]
        return wl.bracketed(
            begin=cls_name, docs=docs, sep=wl.comma, end=wl.TextDoc(")"), indent=4
        )

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


##############################################################################
# AbstractFixedComponentsChart


@no_type_check
def _get_tuple(tp: GAT, /) -> GAT:
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
                cls._components = _get_tuple(args[0])  # type: ignore[attr-defined]
                cls._coord_dimensions = _get_tuple(args[1])  # type: ignore[attr-defined]
                break

        super().__init_subclass__(**kw)  # AbstractChart has.

    @property
    def components(self) -> Ks:
        return self._components  # type: ignore[attr-defined]

    @property
    def coord_dimensions(self) -> Ds:
        return self._coord_dimensions  # type: ignore[attr-defined]


@jtu.register_static
class AbstractDimensionalFlag:
    """Marker base class for dimension *flags*.

    A dimension flag is a lightweight mixin used for typing and dispatch. Flags
    do not store data; instead, they classify a chart. These flags must be
    combined with concrete subclasses of :class:`AbstractChart`.


    """

    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        if n is not None:
            DIMENSIONAL_FLAGS[n] = cls

        # Enforce that this is a subclass of AbstractChart unless it's an
        # abstract base class (name starts with "Abstract")
        if not cls.__name__.startswith("Abstract") and not issubclass(
            cls, AbstractChart
        ):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        # Call super() if it defines __init_subclass__
        if callable(super_init_subclass := getattr(super(), "__init_subclass__", None)):
            super_init_subclass(**kw)


DIMENSIONAL_FLAGS: Final[dict[int | L["N"], type[AbstractDimensionalFlag]]] = {}
