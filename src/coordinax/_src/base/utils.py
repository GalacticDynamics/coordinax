"""Utilities shared by the base classes."""

__all__: tuple[str, ...] = ()

import inspect


def is_abstract_class(cls: type, /) -> bool:
    """Determine if a class is abstract.

    A class counts as abstract when it has unimplemented abstract methods, or
    -- by this package's naming convention -- when its name starts with
    ``Abstract``.

    >>> import coordinax.charts as cxc
    >>> from coordinax._src.base.utils import is_abstract_class

    >>> is_abstract_class(cxc.AbstractChart)
    True

    >>> is_abstract_class(cxc.Cart3D)
    False

    """
    return inspect.isabstract(cls) or cls.__name__.startswith("Abstract")
