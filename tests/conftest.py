"""Test configuration for coordinax tests."""

__all__ = ("POSITION_CLASSES", "VELOCITY_CLASSES", "ACCELERATION_CLASSES")

import inspect

from typing import Final

import coordinax as cx


def build_subclass_tuple(
    base_class: type, /, *, exclude_abstract: bool = True
) -> tuple[type, ...]:
    """Build a set of all subclasses of a given base class.

    Parameters
    ----------
    base_class : type
        The base class to find subclasses of.
    exclude_abstract : bool, optional
        Whether to exclude abstract subclasses, by default True.

    Returns
    -------
    tuple[type, ...]
        A tuple of all subclasses of the base class.

    """
    subclasses = []

    def recurse(cls: type) -> None:
        for subclass in cls.__subclasses__():
            if exclude_abstract and inspect.isabstract(subclass):
                recurse(subclass)
            else:
                subclasses.append(subclass)
                recurse(subclass)

    recurse(base_class)
    return tuple(subclasses)


POSITION_CLASSES: Final = build_subclass_tuple(cx.r.AbstractPos, exclude_abstract=True)
VELOCITY_CLASSES: Final = build_subclass_tuple(cx.r.AbstractVel, exclude_abstract=True)
ACCELERATION_CLASSES: Final = build_subclass_tuple(
    cx.r.AbstractAcc, exclude_abstract=True
)
