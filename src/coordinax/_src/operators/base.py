"""Base classes for operators on coordinates."""

__all__ = ("AbstractOperator",)

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, final

import equinox as eqx
import jax.numpy as jnp
import wadler_lindig as wl
from jaxtyping import ArrayLike
from plum import dispatch

import unxt as u
from dataclassish import field_items, flags

from .api import operate, simplify

if TYPE_CHECKING:
    import coordinax.ops  # noqa: ICN001

_sentinel: Final = object()


@final
@dataclass(slots=True)
class Neg:
    """A parameter that negates another parameter."""

    param: Any
    """The parameter to negate."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the parameter and negate the result."""
        return -self.param(*args, **kwargs)

    def __neg__(self) -> Any:
        """Return the original parameter."""
        return self.param


class AbstractOperator(eqx.Module):
    """Abstract base class for operators on coordinates.

    An operator is an object that defines a transformation on coordinates. It
    can be applied to a set of coordinates to produce a new set of coordinates.
    Operators can be composed together to form a sequence of transformations.

    """

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractOperator]", *args: object, **kwargs: object
    ) -> "AbstractOperator":
        """Construct from a set of arguments."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Operator API

    @dispatch
    def __call__(self: "AbstractOperator", tau: Any, x: Any, /, **kwargs: Any) -> Any:
        """Apply the operator to the arguments.

        This method calls `coordinax.ops.operate` to apply the operator.

        """
        return operate(self, tau, x, **kwargs)

    @dispatch
    def __call__(self: "AbstractOperator", x: Any, /, **kwargs: Any) -> Any:
        """Apply the operator to the arguments.

        This method calls `coordinax.ops.operate` to apply the operator.

        """
        return operate(self, None, x, **kwargs)

    @classmethod
    @abstractmethod
    def operate(cls, params: dict[str, Any], x: Any, /) -> Any:
        """Apply the operator to the input."""
        raise NotImplementedError  # pragma: no cover

    # -------------------------------------------

    @property
    @abstractmethod
    def inverse(self) -> "AbstractOperator":
        """The inverse of the operator."""
        ...

    def simplify(self) -> "AbstractOperator":
        """Simplify the operator.

        This method calls `coordinax.ops.simplify` to simplify the
        operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.ops as cxo

        >>> op = cxo.Identity()
        >>> op.simplify() is op
        True

        >>> pipe = cxo.Pipe((cxo.Identity(), cxo.Identity()))
        >>> pipe
        Pipe((Identity(), Identity()))
        >>> pipe.simplify()
        Identity()

        >>> op = cxo.GalileanOp(translation=u.Quantity([0., 2., 3., 4.], "km"))
        >>> op.simplify()
        Add(delta=Quantity(f32[4], unit='km'))

        """
        return simplify(self)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the documentation for the operator.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the documentation
            generation functions.

            - compact_arrays : bool, optional
            - indent : int, optional

        """
        # Get the field items, excluding those that should not be shown and
        # those that are equal to the default value.
        fitems = list(field_items(flags.FilterRepr, self))
        fitems = [
            (k, v)
            for k, v in fitems
            if not jnp.all(v == getattr(self.__class__, k, _sentinel))
        ]
        # Make the field docs list
        if len(fitems) == 1 and kwargs.get("compact_arrays", False):
            docs = [wl.TextDoc(str(fitems[0][1]))]
        else:
            docs = wl.named_objs(fitems, **kwargs)

        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs.get("indent", 4),
        )

    # ===============================================================
    # Python API

    def __str__(self) -> str:
        """Return a string representation of the operator.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> print(op)
        Rotate([[1 0 0]
                          [0 1 0]
                          [0 0 1]])

        >>> op = cx.ops.GalileanOp(
        ...     translation=u.Quantity([0., 2, 3, 4], "km"),
        ...     velocity=u.Quantity([1., 2, 3], "km/s"),
        ...     rotation=jnp.eye(3).at[0, 2].set(1),
        ... )
        >>> print(op)
        GalileanOp(
            rotation=Rotate([[1. 0. 1.]
                            [0. 1. 0.]
                            [0. 0. 1.]]),
            translation=Add(Quantity['length']([0., 2., 3., 4.], unit='km')),
            velocity=Add(Quantity['speed']([1., 2., 3.], unit='km / s'))
        )

        """
        return wl.pformat(
            self, width=88, vector_form=True, short_arrays=False, compact_arrays=True
        )

    # -------------------------------------------
    # Operator Composition

    def __or__(self, other: "AbstractOperator") -> "coordinax.ops.Pipe":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax.ops as cxo

        >>> op1 = cxo.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> op2 = cxo.Identity()
        >>> op3 = op1 | op2
        >>> op3
        Pipe((Rotate(R=i32[3,3]), Identity()))

        >>> op4 = cxo.Identity() | op3
        >>> op4
        Pipe((Identity(), Rotate(R=i32[3,3]), Identity()))

        """
        from .pipe import Pipe  # noqa: PLC0415

        # Defer to other's __ror__ if it's a Pipe
        if isinstance(other, Pipe):
            return other.__ror__(self)

        # Otherwise, create a new Pipe
        return Pipe((self, other))


# ============================================================
# Constructors


@AbstractOperator.from_.dispatch(precedence=-1)
def from_(
    cls: type[AbstractOperator], *args: object, **kwargs: object
) -> AbstractOperator:
    """Construct from a set of arguments.

    This is a low-priority dispatch that will be called if no other
    dispatch is found. It just tries to pass the arguments to the
    constructor.

    """
    return cls(*args, **kwargs)


@AbstractOperator.from_.dispatch
def from_(cls: type[AbstractOperator], obj: Mapping[str, Any], /) -> AbstractOperator:
    """Construct from a mapping.

    Examples
    --------
    >>> import coordinax.ops as cxo
    >>> cxo.Pipe.from_({"operators": (cxo.Identity(), cxo.Identity())})
    Pipe((Identity(), Identity()))

    """
    return cls(**obj)


@AbstractOperator.from_.dispatch
def from_(
    cls: type[AbstractOperator],
    x: ArrayLike | list[float | int],
    unit: str,  # TODO: support unit object
    /,
) -> AbstractOperator:
    """Construct from a Quantity's value and unit.

    Examples
    --------
    >>> import coordinax.ops as cxo

    >>> op = cxo.GalileanOp.from_([1, 1, 1], "km")
    >>> print(op)
    GalileanOp(<CartesianPos3D: (x, y, z) [km]
        [1 1 1]>)

    >>> op = cxo.Add.from_([3e5, 1, 1, 1], "km")
    >>> print(op)
    Add(
      delta_t=Quantity(1.0006922, unit='s'),
      delta_q=<CartesianPos3D: (x, y, z) [km]
          [1. 1. 1.]>
    )

    >>> op = cxo.Add.from_([1, 1, 1], "km/s")
    >>> print(op)
    Add(<CartesianVel3D: (x, y, z) [km / s]
        [1 1 1]>)

    """
    return cls.from_(u.Quantity(x, unit))


@AbstractOperator.from_.dispatch
def from_(cls: type[AbstractOperator], obj: AbstractOperator, /) -> AbstractOperator:
    """Construct an operator from another operator.

    Raises
    ------
    TypeError
        If the input object is not a subclass of the target class.

    Examples
    --------
    >>> import coordinax as cx

    If the object is the same type, it should return the object itself.

    >>> op = cx.ops.Identity()
    >>> cx.ops.Identity.from_(op) is op
    True

    If the object is a different type, it will error.

    >>> try:
    ...     cx.ops.Add.from_(op)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct <class 'coordinax...Add'> from <class 'coordinax...Identity'>.

    Unless the object is a subclass of the target class.

    >>> class MyOperator(cx.ops.Identity):
    ...     pass

    >>> op = MyOperator()
    >>> op
    MyOperator()

    >>> newop = cx.ops.Identity.from_(op)
    >>> newop is op, isinstance(newop, cx.ops.Identity)
    (False, True)

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. `isinstance` is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))
