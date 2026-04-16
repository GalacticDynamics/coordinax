"""Wrapping/decorator utilities."""

__all__ = ("strip_return_annotation",)

import inspect
from typing import Any, Final, TypeVar

T = TypeVar("T")
EMPTY: Final = inspect.Parameter.empty


def strip_return_annotation(func: T, /) -> T:
    """Remove the return type annotation from a function and any wrapped functions.

    This utility helps work around Plum dispatch issues where parameterized
    generic return types (like ``SearchStrategy[SomeType]``) cause type
    resolution errors during method registration.

    Works with functions wrapped by ``@st.composite``, which sets both
    ``__annotations__`` and ``__signature__`` on the wrapper.

    Parameters
    ----------
    func
        The function to modify in place.

    Returns
    -------
    Callable
        The same function with its return annotation removed.

    Examples
    --------
    >>> from coordinax.hypothesis.utils import strip_return_annotation
    >>> import hypothesis.strategies as st

    Use with functions decorated by multiple dispatches and Hypothesis:

    >>> @strip_return_annotation
    ... @st.composite
    ... def my_strategy(draw) -> str:
    ...     return draw(st.just("hello"))

    """
    # Strip from __annotations__
    if hasattr(func, "__annotations__") and "return" in func.__annotations__:
        del func.__annotations__["return"]

    # @st.composite also sets __signature__ explicitly; Plum reads that first
    f: Any = func
    if hasattr(f, "__signature__"):
        sig: inspect.Signature = f.__signature__  # type: ignore[assignment]
        if sig.return_annotation is not EMPTY:
            f.__signature__ = sig.replace(return_annotation=EMPTY)  # type: ignore[assignment]

    # Recurse into wrapped functions (in case applied before @st.composite)
    if hasattr(func, "__wrapped__"):
        strip_return_annotation(func.__wrapped__)

    return func
