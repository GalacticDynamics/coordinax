"""Module for registering operator APIs."""

__all__: tuple[str, ...] = ()

from typing import Any

import equinox as eqx
import jax.tree as jtu
from plum import dispatch

import unxt as u

from .base import AbstractOperator


def parameters_dict(op: AbstractOperator, /) -> dict[str, Any]:
    """Get the parameters of an operator as a dictionary."""
    return {k: getattr(op, k) for k in op.__dataclass_fields__}


@dispatch
def operate(op: AbstractOperator, tau: Any, /, *xs: Any, **kw: Any) -> Any:
    """Apply an operator with a time argument.

    Examples
    --------
    >>> import coordinax.ops as cxo

    >>> add = cxo.Add(lambda t: t.ustrip("yr"))
    >>> tau = u.Quantity(5, "yr")
    >>> x = u.Quantity([1, 2, 3], "km")

    >>> cxo.operate(add, tau, x)
    Quantity(Array([6, 7, 8], dtype=int32), unit='km')

    """
    # Partition the operator into static and dynamic parameters. The dynamic
    # parameters are functions of time.
    params = parameters_dict(op)
    dynamic, static = eqx.partition(params, filter_spec=callable)
    # Evaluate the dynamic parameters at the given time.
    eval_dynamic = jtu.map(lambda p: p(tau), dynamic)
    # Recombine the static and evaluated dynamic parameters.
    params = eqx.combine(static, eval_dynamic)
    # Apply the operator with the time-evaluated parameters.
    return operate(type(op), params, *xs, **kw)


@dispatch
def operate(op: type[AbstractOperator], params: dict[str, Any], x: Any, /) -> Any:
    """Apply an operator without a time argument."""
    return op.operate(params, x)
