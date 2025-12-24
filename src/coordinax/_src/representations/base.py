"""Vector."""

__all__ = (
    "AbstractRep",
    "AbstractFixedComponentsRep",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
)

from abc import ABCMeta, abstractmethod

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

import wadler_lindig as wl

import unxt as u

from . import api

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # type: ignore[misc]
Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
V = TypeVar("V")

REPRESENTATION_CLASSES: list[type["AbstractRep[Any, Any]"]] = []


class AbstractRep(Generic[Ks, Ds], metaclass=ABCMeta):
    """Abstract base class for representations of vectors."""

    def __init_subclass__(cls) -> None:
        # This allows multiple inheritance with other ABCs that might or might
        # not define an `__init_subclass__`
        if hasattr(cls, "__init_subclass__"):
            super().__init_subclass__()

        # Register the representation
        REPRESENTATION_CLASSES.append(cls)

    # ===============================================================
    # Vector API

    @property
    @abstractmethod
    def components(self) -> Ks:
        """The names of the components."""
        ...

    @property
    @abstractmethod
    def coord_dimensions(self) -> Ds:
        """The dimensions of the components."""
        ...

    @property
    def dimensionality(self) -> int:
        return len(self.components)

    @property
    def cartesian(self) -> "AbstractRep[Ks, Ds]":
        """Return the corresponding Cartesian vector class."""
        return api.cartesian_rep(self)

    def check_data(self, data: Mapping[str, Any], /) -> None:
        # Check that the keys of data match kind.components
        if set(data.keys()) != set(self.components):
            msg = (
                "Data keys do not match kind components: "
                f"{set(data.keys())} != {set(self.components)}"
            )
            raise ValueError(msg)

        # Check that the dimensions match kind.coord_dimensions
        for v, d in zip(data.values(), self.coord_dimensions, strict=True):
            if d is not None and u.dimension_of(v) != d:
                msg = "Data dimensions do not match"
                raise ValueError(msg)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kw: object) -> wl.AbstractDoc:
        # TODO: vectorform
        return wl.TextDoc(
            f"{self.__class__.__name__}({self.components}, {self.coord_dimensions})"
        )


@no_type_check
def get_tuple(tp: GAT, /) -> GAT:
    return tuple(arg.__args__[0] for arg in get_args(tp))


class AbstractFixedComponentsRep(AbstractRep[Ks, Ds]):
    """Abstract base class for representations with fixed components and dimensions."""

    def __init_subclass__(cls) -> None:
        # Extract Ks and Ds from AbstractFixedComponentsRep in the inheritance
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is AbstractFixedComponentsRep:
                args = get_args(base)
                if len(args) != 2:
                    raise TypeError
                cls._components = get_tuple(args[0])  # type: ignore[attr-defined]
                cls._coord_dimensions = get_tuple(args[1])  # type: ignore[attr-defined]
                break

        super().__init_subclass__()  # AbstractRep has.

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
    :class:`AbstractRep` to define the meaning of a vector.
    """

    def __init_subclass__(cls, n: int | L["N"] | None = None) -> None:
        if n is not None:
            DIMENSIONAL_FLAGS[n] = cls

        # Enforce that this is a subclass of AbstractRep
        if not cls.__name__.startswith("Abstract") and not issubclass(cls, AbstractRep):
            msg = f"{cls.__name__} must be a subclass of AbstractRep"
            raise TypeError(msg)

        if hasattr(super(), "__init_subclass__"):
            super().__init_subclass__()


DIMENSIONAL_FLAGS: Final[dict[int | L["N"], type[AbstractDimensionalFlag]]] = {}
