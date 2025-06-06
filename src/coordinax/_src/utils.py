"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated as Ann, Any, Generic, TypeVar
from typing_extensions import Doc

GetterRetT = TypeVar("GetterRetT")
EnclosingT = TypeVar("EnclosingT")


@dataclass(frozen=True)
class ClassPropertyDescriptor(Generic[EnclosingT, GetterRetT]):
    """Descriptor for class properties.

    Parameters
    ----------
    fget : classmethod | staticmethod
        The class/staticmethod wrapped function to be used as the getter for the
        class-level property.

    Examples
    --------
    >>> from coordinax._src.utils import classproperty
    >>> class MyConstants:
    ...     @classproperty
    ...     def pau(cls) -> float:
    ...         return 4.71
    >>> MyConstants.pau
    4.71

    """

    fget: classmethod | staticmethod  # type: ignore[type-arg]
    """function to be used for getting a class-level attribute value."""

    def __get__(
        self, obj: EnclosingT | None, klass: type[EnclosingT] | None
    ) -> GetterRetT:
        """Get the class-level attribute value.

        Parameters
        ----------
        obj
            The object this descriptor is being accessed from. If ``None`` then
            this is being accessed from the class, not an instance.
        klass
            The class this descriptor is being accessed from. If ``None`` then
            this is being accessed from an instance, not the class.

        Returns
        -------
        GetterRetT
            The class-level attribute value.

        Raises
        ------
        TypeError
            If the descriptor is accessed from neither an instance nor a class.

        Examples
        --------
        >>> from coordinax._src.utils import classproperty
        >>> class MyConstants:
        ...     @classproperty
        ...     def pau(cls) -> float:
        ...         return 4.71
        >>> MyConstants.pau
        4.71

        Class properties can only be accessed from the class itself (or its instances):

        >>> clsprop = ClassPropertyDescriptor(staticmethod(lambda: 10.0))
        >>> try: clsprop.__get__(None, None)
        ... except TypeError as e: print(e)
        Descriptor must be accessed from an instance or class.

        From the class:

        >>> clsprop.__get__(None, MyConstants)
        10.0

        From an instance:

        >>> clsprop.__get__(MyConstants(), None)
        10.0

        """
        # Ensure that the descriptor is accessed from an instance or class
        if obj is None and klass is None:
            msg = "Descriptor must be accessed from an instance or class."
            raise TypeError(msg)

        # Get the class from in its instance.
        if klass is None:
            assert obj is not None  # just for mypy # noqa: S101
            klass = type(obj)

        # Forward the call to the class/staticmethod wrapped function
        return self.fget.__get__(obj, klass)()


def classproperty(
    func: Ann[  # type: ignore[type-arg]  # TODO: can the Any be improved?
        Callable[[type[Any]], GetterRetT] | classmethod | staticmethod,
        Doc("function to be used for getting a class-level attribute value."),
    ],
) -> Ann[ClassPropertyDescriptor[Any, GetterRetT], Doc("class-level property")]:
    """Create a class-level property.

    This is most commonly used as a decorator for a function that returns a
    class-level attribute value.

    Parameters
    ----------
    func
        The function to be used as the getter.

    Returns
    -------
    ClassPropertyDescriptor[EnclosingT, GetterRetT]
        The class-level property.

    Examples
    --------
    >>> from coordinax._src.utils import classproperty
    >>> class MyConstants:
    ...     @classproperty
    ...     def pau(cls) -> float:
    ...         return 4.71
    >>> MyConstants.pau
    4.71

    """
    # Ensure that the function is a class/staticmethod
    fget = func if isinstance(func, classmethod | staticmethod) else classmethod(func)
    # Return the wrapped function as a class-level property
    return ClassPropertyDescriptor[Any, GetterRetT](fget)
