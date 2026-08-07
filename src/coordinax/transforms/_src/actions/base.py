"""Base classes for operators on coordinates."""

__all__ = ("AbstractTransform", "is_time_dependent", "materialize_transform")

import abc
import dataclasses

from collections.abc import Mapping
from jaxtyping import ArrayLike
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

import equinox as eqx
import jax.numpy as jnp
import optype as op
import plum
import wadler_lindig as wl

import unxt as u
from dataclassish import field_items, flags

import coordinaxs.api.transforms as cxfmapi
from coordinax.internal import pos_named_objs

if TYPE_CHECKING:
    import coordinax.transforms  # noqa: ICN001

_DataclassBase = op.dataclasses.HasDataclassFields

_sentinel: Final = object()


class AbstractTransform(eqx.Module):
    """Abstract base class for operators on coordinates.

    An operator is an object that defines a transformation on coordinates. It
    can be applied to a set of coordinates to produce a new set of coordinates.
    Operators can be composed together to form a sequence of transformations.

    """

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @plum.dispatch.abstract
    def from_(
        cls: "type[AbstractTransform]", *args: object, **kwargs: object
    ) -> "AbstractTransform":
        """Construct from a set of arguments."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Operator API

    def __call__(self, tau: Any, x: Any = None, /, **kw: Any) -> Any:
        """Apply the operator to the arguments.

        This method calls `coordinaxs.api.frames.act` to apply the operator.
        If `x` is not provided, `tau` will be passed as `None` to the `act`

        """
        if x is None:
            return cxfmapi.act(self, None, tau, **kw)
        return cxfmapi.act(self, tau, x, **kw)

    # -------------------------------------------

    @property
    @abc.abstractmethod
    def inverse(self) -> "AbstractTransform":
        """The inverse of the operator."""
        raise NotImplementedError  # pragma: no cover

    @property
    def is_time_dependent(self) -> bool:
        """Whether the point action depends on the time parameter tau.

        A declared trait, not a structural scan: subclasses whose point
        action varies with tau (``TimeDep``, ``Boost``) override this to
        `True`; composites (``Composed``) override it to the disjunction of
        their components. The default is `False`.

        Examples
        --------
        >>> import coordinax.transforms as cxfm

        >>> cxfm.Translate.from_([1, 2, 3], "km").is_time_dependent
        False

        """
        return False

    def simplify(self) -> "AbstractTransform":
        """Simplify the operator.

        This method calls `coordinax.ops.simplify` to simplify the
        operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm

        >>> op = cxfm.Identity()
        >>> op.simplify() is op
        True

        >>> pipe = cxfm.Composed((cxfm.Identity(), cxfm.Identity()))
        >>> pipe
        Composed((Identity(), Identity()))
        >>> pipe.simplify()
        Identity()

        """
        return cast("AbstractTransform", cxfmapi.simplify(self))

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
            - positional_fields : bool, optional
                If True, introspect the dataclass to display non-keyword-only
                fields without their parameter names (positionally).

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kwargs.setdefault("short_arrays", "compact")
        kwargs.setdefault("use_short_name", True)
        kwargs.setdefault("named_unit", False)
        kwargs.setdefault("include_params", False)

        # Get the field items, excluding those that should not be shown and
        # those that are equal to the default value.
        fitems = list(field_items(flags.FilterRepr, self))
        fitems = [
            (k, v)
            for k, v in fitems  # ty: ignore[not-iterable]
            if not jnp.all(v == getattr(self.__class__, k, _sentinel))
        ]

        # Handle positional fields (fields without names)
        use_positional = kwargs.pop("positional_fields", False)

        # Make the field docs list
        if len(fitems) == 1 and kwargs.get("short_arrays") == "compact":
            docs = [wl.TextDoc(str(fitems[0][1]))]
        elif use_positional:
            # Introspect to find which fields are keyword-only
            kw_only_fields = {f.name for f in dataclasses.fields(self) if f.kw_only}

            # Get positional field names (fields that are not keyword-only)
            pos_names = [k for k, v in fitems if k not in kw_only_fields]

            docs = pos_named_objs(
                fitems, pos_names, self.__dataclass_fields__, **kwargs
            )
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

    def __repr__(self) -> str:
        return wl.pformat(self, use_short_name=True, positional_fields=True)

    def __str__(self) -> str:
        """Return a string representation of the operator.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import coordinax as cx

        >>> op = cxfm.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> print(op)
        Rotate([[1 0 0]
                [0 1 0]
                [0 0 1]])

        """
        return wl.pformat(self, width=88, vector_form=True, short_arrays="compact")

    # -------------------------------------------
    # Operator Composition

    def __or__(self, other: "AbstractTransform", /) -> "coordinax.transforms.Composed":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax.transforms as cxfm

        >>> op1 = cxfm.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> op2 = cxfm.Identity()
        >>> op3 = op1 | op2
        >>> op3
        Composed((Rotate(i64[3,3](jax)), Identity()))

        >>> op4 = cxfm.Identity() | op3
        >>> op4
        Composed((Identity(), Rotate(i64[3,3](jax)), Identity()))

        """
        from .composed import Composed  # noqa: PLC0415

        # Defer to other's __ror__ if it's a Composed
        if isinstance(other, Composed):
            return other.__ror__(self)

        # Otherwise, create a new Composed
        return Composed((self, other))


# ============================================================
# Constructors


@AbstractTransform.from_.dispatch(precedence=-1)  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[AbstractTransform], *args: object, **kwargs: object
) -> AbstractTransform:
    """Construct from a set of arguments.

    This is a low-priority dispatch that will be called if no other
    dispatch is found. It just tries to pass the arguments to the
    constructor.

    """
    return cls(*args, **kwargs)


@AbstractTransform.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractTransform], obj: Mapping[str, Any], /) -> AbstractTransform:
    """Construct from a mapping.

    >>> import coordinax.transforms as cxfm
    >>> cxfm.Composed.from_({"transforms": (cxfm.Identity(), cxfm.Identity())})
    Composed((Identity(), Identity()))

    """
    return cls(**obj)


@AbstractTransform.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[AbstractTransform],
    x: ArrayLike | list[float | int],
    unit: str,  # TODO: support unit object
    /,
) -> AbstractTransform:
    """Construct from a Quantity's value and unit.

    >>> import coordinax.transforms as cxfm
    >>> op = cxfm.Translate.from_([1, 1, 1], "km")
    >>> print(op)
    Translate(
        {'x': Q(1, 'km'), 'y': Q(1, 'km'), 'z': Q(1, 'km')}, chart=Cart3D(M=Rn(3))
    )

    """
    return cls.from_(u.Q(x, unit))  # ty: ignore[invalid-return-type]


@AbstractTransform.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractTransform], obj: AbstractTransform, /) -> AbstractTransform:
    """Construct an operator from another operator.

    Raises
    ------
    TypeError
        If the input object is not a subclass of the target class.

    Examples
    --------
    >>> import coordinax as cx

    If the object is the same type, it should return the object itself.

    >>> op = cxfm.Identity()
    >>> cxfm.Identity.from_(op) is op
    True

    If the object is a different type, it will error.

    >>> try:
    ...     cxfm.Translate.from_(op)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct <class '...Translate'> from <class '...Identity'>.

    Unless the object is a subclass of the target class.

    >>> class MyOperator(cxfm.Identity):
    ...     pass

    >>> op = MyOperator()
    >>> op
    MyOperator()

    >>> newop = cxfm.Identity.from_(op)
    >>> newop is op, isinstance(newop, cxfm.Identity)
    (False, True)

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. `isinstance` is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))  # ty: ignore[no-matching-overload]


# =============================================================================
# materialize_transform: Materialization of time-dependent parameters

# NB: kept as a module-level TypeVar rather than PEP 695 syntax (UP047): the
# inlined type parameter renders as an unresolved `py:class` cross-reference in
# the Sphinx (`-W`) docs build.
OpT = TypeVar("OpT", bound=_DataclassBase)


def materialize_transform(op: OpT, tau: Any, /) -> OpT:  # noqa: UP047
    r"""Evaluate the parametric parts of an operator at a given time.

    This function materializes an operator by evaluating every `TimeDep`
    part of it — recursively, through `Composed` — at the specified time
    ``tau``, returning a new operator with no remaining `TimeDep` parts.

    Mathematically, if an operator $\mathrm{Op}$ has parts that depend on
    an affine parameter $\tau$, then:

    $$
    \mathrm{materialize\_transform}(\mathrm{Op}, \tau) \to \mathrm{Op}_\tau
    $$

    where $\mathrm{Op}_\tau$ is the operator with all `TimeDep` parts
    evaluated at $\tau$.

    Parameters
    ----------
    op : AbstractTransform
        The operator to evaluate. May contain `TimeDep` parts, either
        directly or as `Composed` components.
    tau : Any
        The time/affine parameter at which to evaluate `TimeDep` parts.
        Typically a ``unxt.Quantity`` with time units.

    Returns
    -------
    AbstractTransform
        The operator with every `TimeDep` part evaluated at ``tau``. An
        operator with no `TimeDep` parts is returned unchanged (the same
        object).

    Notes
    -----
    This function is:

    - **Pure**: No side effects, safe for JAX tracing
    - **Recursive**: Descends into `Composed` so nested `TimeDep` parts
      are also evaluated

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    **Time-dependent operator:**

    >>> def build(tau):
    ...     v = jnp.array([1.0, 0.0, 0.0]) * tau.ustrip("s")
    ...     return cxfm.Translate.from_(v, "km")

    >>> op = cxfm.TimeDep.from_(build)
    >>> tau = u.Q(5.0, "s")
    >>> op_eval = cxfm.materialize_transform(op, tau)
    >>> op_eval.delta["x"]
    Q(5., 'km')

    **Static operator (no change):**

    >>> op_static = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.materialize_transform(op_static, tau) is op_static
    True

    **Identity operator:**

    >>> identity = cxfm.Identity()
    >>> cxfm.materialize_transform(identity, tau)
    Identity()

    See Also
    --------
    act : Apply an operator to an input (calls ``materialize_transform`` internally)

    """
    # Local imports to avoid a cycle (timedep.py imports base.py).
    from .composite import AbstractCompositeTransform  # noqa: PLC0415
    from .timedep import TimeDep  # noqa: PLC0415

    if isinstance(op, TimeDep):
        inner = op.materialize(tau)  # raises TypeError on tau=None
        return cast("OpT", materialize_transform(inner, tau))
    if isinstance(op, AbstractCompositeTransform):
        new = tuple(materialize_transform(t, tau) for t in op.transforms)
        return cast("OpT", dataclasses.replace(op, transforms=new))
    return op


def is_time_dependent(op: Any, /) -> bool:
    r"""Whether the operator's point action depends on the time parameter tau.

    This is a declared trait — `AbstractTransform.is_time_dependent` — not a
    structural scan: it delegates to the operator's own property, which
    subclasses override where their point action is intrinsically
    tau-dependent (`TimeDep`, `Boost`) or is a disjunction of their
    components' (`Composed`).

    Parameters
    ----------
    op
        The operator to inspect.

    Returns
    -------
    bool
        A plain Python bool, safe to branch on during JAX tracing.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> static = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.is_time_dependent(static)
    False

    >>> moving = cxfm.TimeDep.from_(
    ...     lambda t: cxfm.Translate.from_(
    ...         jnp.asarray([1.0, 0.0, 0.0]) * u.ustrip("s", t), "km"
    ...     )
    ... )
    >>> cxfm.is_time_dependent(moving)
    True

    A composition is time-dependent iff any component is:

    >>> cxfm.is_time_dependent(static | moving)
    True
    >>> cxfm.is_time_dependent(static | cxfm.Identity())
    False

    """
    if not isinstance(op, AbstractTransform):
        msg = f"is_time_dependent expects an AbstractTransform, got {type(op)}"
        raise TypeError(msg)
    return op.is_time_dependent
