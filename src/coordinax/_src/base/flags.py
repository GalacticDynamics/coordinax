"""Flag types on which to dispatch."""

__all__: list[str] = []

from typing import Any, NoReturn, final

from dataclassish.flags import FlagConstructionError


@final
class AttrFilter:
    """Flag to filter out `VectorAttribute` fields.

    Examples
    --------
    >>> import dataclasses
    >>> from dataclassish import field_items
    >>> from coordinax import VectorAttribute, AttrFilter, AbstractPos

    >>> class TestPos(AbstractPos):
    ...    x: int
    ...    attr: float = VectorAttribute(default=2.0)

    >>> obj = TestPos(x=1)

    >>> [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    [('x', 1), ('attr', 2.0)]

    >>> field_items(obj)
    (('x', 1), ('attr', 2.0))

    >>> field_items(AttrFilter, obj)
    (('x', 1),)

    """

    def __new__(cls, *_: Any, **__: Any) -> NoReturn:
        raise FlagConstructionError(flag_type="AttrFilter")