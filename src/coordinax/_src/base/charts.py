"""Charts."""

__all__ = (
    "AbstractChart",
    "AbstractStaticChart",
    "AbstractParameterizedChart",
    "AbstractFixedComponentsChart",
    "AbstractStaticFixedComponentsChart",
    "AbstractParameterizedFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "CHART_CLASSES",
    "NON_ABC_CHART_CLASSES",
    "chart_dataclass_decorator",
    "is_not_abstract_chart_subclass",
    "MISSING",
    "CDictT",
    "MT",
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

import equinox as eqx
import jax.tree_util as jtu
import plum
import wadler_lindig as wl

import dataclassish
import unxt as u

from .manifold import AbstractManifold
from coordinax._src.custom_types import CDictT, Ds, Ks

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # ty: ignore[invalid-type-form]
MT = TypeVar("MT", bound=AbstractManifold)
V = TypeVar("V")

# Charts are registered in CHART_CLASSES when they are defined, via
# AbstractChart.__init_subclass__. This allows us to find all chart classes for
# dispatch and other purposes. We use a weak set to avoid keeping classes alive
# unnecessarily, and a mapping proxy to prevent modification of the set from
# outside this module.
CHART_CLASSES: weakref.WeakSet[type["AbstractChart[AbstractManifold, Any, Any]"]] = (
    weakref.WeakSet()
)

NON_ABC_CHART_CLASSES: weakref.WeakSet[
    type["AbstractChart[AbstractManifold, Any, Any]"]
] = weakref.WeakSet()

chart_dataclass_decorator = dataclasses.dataclass(
    frozen=True, slots=False, repr=False, eq=False
)

MISSING = object()


##############################################################################
# AbstractChart


def _field_values(chart: "AbstractChart[Any, Any, Any]", /) -> tuple[Any, ...]:
    """Field values, as `dataclassish.field_values` gives them.

    Direct walk when the chart is a dataclass -- as all built-in ones are --
    since the plum dispatch costs ~25x. Other charts take the general path.
    """
    if not dataclasses.is_dataclass(chart):
        return tuple(dataclassish.field_values(chart))
    return tuple(getattr(chart, f.name) for f in dataclasses.fields(chart))


def _is_dynamic(value: Any, /) -> bool:
    """Whether a field value can hold an array or tracer.

    A `unxt.StaticQuantity` contributes no pytree leaves and is static; a
    `unxt.Quantity` contributes one array leaf and is dynamic. Values that are
    not registered pytrees at all are a single *non-array* leaf, and stay
    static -- they compare and hash as before.
    """
    return any(eqx.is_array(x) for x in jtu.tree_leaves(value))


def _check_on_exactly_one_branch(cls: type, /) -> None:
    """Reject a concrete chart that is not on exactly one branch.

    A chart on neither branch is never registered static and silently becomes
    one opaque pytree leaf; a chart on both is a contradiction. Neither fails
    loudly on its own, so fail here, at class creation.

    The branch classes are resolved from module globals because they are defined
    *below* `AbstractChart` -- while they are themselves being created the names
    are absent, and abstract classes are exempt anyway.
    """
    static = globals().get("AbstractStaticChart")
    param = globals().get("AbstractParameterizedChart")
    if static is None or param is None:
        return
    on_static, on_param = issubclass(cls, static), issubclass(cls, param)
    if on_static == on_param:
        msg = (
            f"{cls.__name__} is on {'both' if on_static else 'neither'} chart "
            "branch; a concrete chart must subclass exactly one of "
            "AbstractStaticChart, AbstractParameterizedChart"
        )
        raise TypeError(msg)


def _static_field_values(chart: "AbstractChart[Any, Any, Any]", /) -> tuple[Any, ...]:
    """Field values that hold no arrays or tracers, in field order."""
    return tuple(v for v in _field_values(chart) if not _is_dynamic(v))


class AbstractChart(Generic[MT, Ks, Ds], metaclass=abc.ABCMeta):
    """Abstract base class for charts (coordinate representations)."""

    M: MT
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
            _check_on_exactly_one_branch(cls)

    # ===============================================================
    # Vector API

    @property
    @abc.abstractmethod
    def components(self) -> Ks:
        """The names of the components."""
        raise NotImplementedError  # pragma: no cover

    @property
    @abc.abstractmethod
    def coord_dimensions(self) -> Ds:
        """The dimensions of the components."""
        raise NotImplementedError  # pragma: no cover

    @property
    def ndim(self) -> int:
        """Number of coordinate components (chart dimension)."""
        return len(self.components)

    @property
    @abc.abstractmethod
    def cartesian(self) -> "AbstractChart[MT, Ks, Ds]":
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
            Delta=StaticQuantity(i64[](numpy), unit='km'), M=Rn(3)
        )

        >>> wl.pprint(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(20, "km")),
        ... short_arrays=False)
        ProlateSpheroidal3D[('mu', 'nu', 'phi'), ('area', 'area', 'angle')](
            Delta=StaticQuantity(array(20), unit='km'), M=Rn(3)
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
            or v is not (defaults[k].default if k in defaults else MISSING)
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

        Two charts are equal only when provably so: the same object, or the
        same type with no dynamic (traced/array) parameters on either side and
        equal static fields. This is deliberately conservative -- a rule that
        inspected dynamic values would behave differently inside and outside
        `jax.jit`.

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
        if self is other:
            return True
        # Conservative: a dynamic parameter cannot be compared at trace time.
        # Not provably equal => not equal.
        assert isinstance(other, AbstractChart)  # noqa: S101  # for mypy
        self_values, other_values = _field_values(self), _field_values(other)
        if any(map(_is_dynamic, self_values)) or any(map(_is_dynamic, other_values)):
            return False
        # Neither side has a dynamic field, so these *are* the static values.
        # Check the components, coord_dimensions, and static fields for equality
        return (
            self.components == other.components
            and self.coord_dimensions == other.coord_dimensions
            and self_values == other_values
        )

    def __hash__(self) -> int:
        """Hash a chart based on its type and static field values.

        Dynamic (traced/array) parameters are excluded: they are unhashable,
        and `__eq__` never uses them either, so equal charts still hash equal.

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
                _static_field_values(self),
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
# The two branches of the chart hierarchy.
# NOTE: these must be defined after `is_abstract_class`, which
# `__init_subclass__` calls while the class body is being created.


class AbstractStaticChart(AbstractChart[MT, Ks, Ds]):
    """A chart with no parameters, and therefore no pytree leaves.

    Concrete subclasses are registered as static automatically -- staticness
    is structural, inherited from this branch, not a decorator that can be
    forgotten. (`jtu.register_static` is not inherited, so an unregistered
    chart would silently become a single opaque leaf.)
    """

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # NOTE: static charts must use `chart_dataclass_decorator` (slots=False).
        # `dataclass(slots=True)` builds a second class object and would register
        # both it and the discarded first one.
        if not is_abstract_class(cls):
            jtu.register_static(cls)

    def __post_init__(self) -> None:
        """Reject an array (or tracer) hiding inside a static chart.

        `jtu.register_static` makes the whole instance one static node, so an
        array held in a field reports *zero* pytree leaves: `jit` bakes it in as
        a constant and a tracer walks straight out through the boundary, to die
        later somewhere unrelated. The annotations that would forbid this
        (`GalileanCT.spatial_chart`, `CartesianProductChart.factors`) are not
        enforced at runtime, so enforce it here, at construction.

        Subclasses that define their own `__post_init__` must call
        `super().__post_init__()`.

        This runs on every static-chart construction, including hot ones like
        `chart.cartesian`, so it walks the fields directly rather than through
        the (plum-dispatched, ~50x more expensive) `dataclassish.field_items`,
        and takes a single pass over all of them. Naming the offending fields
        costs a second, per-field pass -- paid only when raising.

        `jtu.tree_leaves` stops at anything that is not a *registered* pytree, so
        a live array hidden inside one would still slip through as a single
        opaque non-array leaf. `AbstractEmbeddingMap` was the one such holder in
        the codebase; it is now a pytree, and its chart holder `EmbeddedChart`
        is on the parameterized branch.
        """
        fields = dataclasses.fields(self)  # ty: ignore[invalid-argument-type]
        values = [getattr(self, f.name) for f in fields]
        if not any(eqx.is_array(x) for x in jtu.tree_leaves(values)):
            return
        bad = sorted(f.name for f in fields if _is_dynamic(getattr(self, f.name)))
        msg = (
            f"{type(self).__name__} is a static chart, but {bad} hold arrays. "
            "A chart parameter that can hold an array must live on a "
            "parameterized chart (`AbstractParameterizedChart`); a static chart "
            "would hide it from JAX entirely."
        )
        raise TypeError(msg)


class AbstractParameterizedChart(AbstractChart[MT, Ks, Ds], eqx.Module):
    """A chart carrying parameters, and therefore a pytree.

    Whether an instance actually has leaves depends on what it is given: a
    `unxt.StaticQuantity` parameter contributes none (hashable, behaves like a
    static chart), a `unxt.Quantity` contributes one (differentiable).
    """


##############################################################################
# AbstractFixedComponentsChart


@no_type_check
def _get_tuple(tp: GAT, /) -> GAT:  # noqa: UP047
    return tuple(arg.__args__[0] for arg in get_args(tp))


class AbstractFixedComponentsChart(AbstractChart[MT, Ks, Ds]):
    """Abstract base class for charts with fixed components and dimensions.

    Having fixed components is orthogonal to the static/parameterized split, so
    this sits *above* it. Concrete charts inherit a branch-bound subclass:
    `AbstractStaticFixedComponentsChart` or
    `AbstractParameterizedFixedComponentsChart`.
    """

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
                    if len(args) != 3:
                        raise TypeError
                    cls._components = _get_tuple(args[1])
                    cls._coord_dimensions = _get_tuple(args[2])
                    break

            # Check the component count matches the declared dimension flag,
            # but only when the chart mixes in an `AbstractDimensionalFlag` with
            # a fixed integer `n` (skip the variable-`n` flag, e.g. `CartND`
            # whose `_chart_ndim` is `"N"`, and charts with no flag at all).
            ndim = getattr(cls, "_chart_ndim", None)
            if (
                isinstance(ndim, int)
                and hasattr(cls, "_components")
                and len(cls._components) != ndim
            ):
                msg = (
                    f"{cls.__name__} is declared {ndim}D but has "
                    f"{len(cls._components)} components {cls._components}"
                )
                raise TypeError(msg)

        super().__init_subclass__(**kw)  # the branch base registers `cls`.

    @property
    def components(self) -> Ks:
        return self._components

    @property
    def coord_dimensions(self) -> Ds:
        return self._coord_dimensions


class AbstractStaticFixedComponentsChart(
    AbstractFixedComponentsChart[MT, Ks, Ds], AbstractStaticChart[MT, Ks, Ds]
):
    """Fixed-components chart with no parameters (the common case)."""


class AbstractParameterizedFixedComponentsChart(
    AbstractFixedComponentsChart[MT, Ks, Ds], AbstractParameterizedChart[MT, Ks, Ds]
):
    """Fixed-components chart carrying parameters, e.g. `ProlateSpheroidal3D`."""


##############################################################################


@jtu.register_static
class AbstractDimensionalFlag:
    """Marker base class for dimension *flags*.

    A dimension flag is a lightweight mixin used for typing and dispatch. Flags
    do not store data; instead, they classify a chart. These flags must be
    combined with concrete subclasses of {class}`AbstractChart`.


    """

    #: Declared coordinate dimension of the flag (set when ``n`` is given).
    _chart_ndim: ClassVar[int | L["N"]]

    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        if n is not None:
            DIMENSIONAL_FLAGS[n] = cls
            # Record the declared dimension so concrete fixed-component charts
            # can validate their component count against it.
            cls._chart_ndim = n

        # Enforce that this is a subclass of AbstractChart unless it's an
        # abstract base class (name starts with "Abstract")
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        # Call super() if it defines __init_subclass__
        if callable(super_init_subclass := getattr(super(), "__init_subclass__", None)):
            super_init_subclass(**kw)


DIMENSIONAL_FLAGS: Final[dict[int | L["N"], type[AbstractDimensionalFlag]]] = {}
