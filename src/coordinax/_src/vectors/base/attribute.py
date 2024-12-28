"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field, is_dataclass
from enum import Enum, auto
from typing import Any, Generic, TypeVar, Union, final, overload

from .base import AbstractVector

Return = TypeVar("Return")


class Sentinel(Enum):
    """Sentinel values for VectorAttribute fields."""

    MISSING = auto()
    """A sentinel value signifying a missing default."""

    def __repr__(self) -> str:
        return f"<{self.name}>"


@final
@dataclass(frozen=True, slots=True)
class VectorAttribute(Generic[Return]):
    """Descriptor for attributes (non-coordinate fields) on a vector.

    Examples
    --------
    >>> import coordinax as cx

    >>> class TestPos(cx.vecs.AbstractPos):
    ...     x: int
    ...     attr: float = cx.vecs.VectorAttribute(default=2.0)
    ...     _dimensionality = 1

    vector-attributes are used to define fields on a vector that are not one of
    the coordinates.

    >>> obj = TestPos(x=1)
    >>> obj.components
    ('x',)

    """

    _: KW_ONLY

    default: Any = Sentinel.MISSING
    """Default value of the Parameter.

    By default set to ``MISSING``, which indicates the attribute must be set
    when initializing the vector object.
    """

    converter: Callable[[Any], Return] = lambda x: x
    """Function to convert the input value to the desired type."""

    name: str = field(
        init=False, compare=True, default="name not initialized", repr=False
    )
    """The name of the attribute on the Vector.

    Cannot be set directly. Set by the container class when the attribute is
    initialized.
    """

    def __post_init__(self) -> None:
        if self.default is not Sentinel.MISSING:
            object.__setattr__(self, "default", self.converter(self.default))

        self.__set_name__(type(self), "name not initialized")

    def __set_name__(self, ref_cls: type, name: str) -> None:
        # attribute name on container cosmology class
        object.__setattr__(self, "name", name)

    @overload
    def __get__(
        self, obj: None, obj_cls: type[AbstractVector]
    ) -> "VectorAttribute[Return]": ...

    @overload
    def __get__(self, obj: AbstractVector, obj_cls: None) -> Return: ...

    def __get__(
        self,
        obj: AbstractVector | None,
        obj_cls: type[AbstractVector] | None,
    ) -> Union["VectorAttribute[Return]", Return]:
        # Get from class
        if obj is None:
            # If the Parameter is being set as part of a dataclass constructor, then we
            # raise an AttributeError if the default is MISSING. This is to prevent the
            # Parameter from being set as the default value of the dataclass field and
            # erroneously included in the class' __init__ signature.
            if self.default is Sentinel.MISSING and (
                not is_dataclass(obj_cls)
                or self.name not in obj_cls.__dataclass_fields__
            ):
                raise AttributeError
            return self
        # Get from instance
        # return getattr(obj, "_" + self.name)
        return obj.__dict__[self.name]

    def __set__(self, obj: AbstractVector, value: Any) -> None:
        if value is self:
            value = self.default

        # Validate value, generally setting units if present
        value = self.converter(value)

        # object.__setattr__(obj, "_" + self.name, value)
        obj.__dict__[self.name] = value
