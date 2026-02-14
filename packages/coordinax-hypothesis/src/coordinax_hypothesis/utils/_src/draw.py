"""Draw Utilities."""

__all__ = ("draw_if_strategy",)


from typing import TypeVar

import hypothesis.strategies as st

T = TypeVar("T")


def draw_if_strategy(draw: st.DrawFn, v: T | st.SearchStrategy[T], /) -> T:
    """Draw from ``v`` if it is a strategy, otherwise return it unchanged.

    This is a convenience helper for functions whose parameters may be either
    concrete values **or** Hypothesis strategies.  When ``v`` is a
    {class}`~hypothesis.strategies.SearchStrategy` it is drawn via *draw*; any
    other value passes through untouched.

    Parameters
    ----------
    draw : ~hypothesis.strategies.DrawFn
        The Hypothesis draw callable (typically from ``@given(st.data())`` or a
        ``@composite`` function).
    v : T | ~hypothesis.strategies.SearchStrategy[T]
        A concrete value or a Hypothesis search strategy.

    Returns
    -------
    T
        The concrete value—either ``v`` itself or a value drawn from it.

    Examples
    --------
    Inside a ``@composite`` strategy the helper removes boilerplate
    ``isinstance`` checks:

    >>> import hypothesis.strategies as st
    >>> from coordinax_hypothesis.utils import draw_if_strategy

    With a plain value, the value is returned as-is:

    >>> data = st.DataObject(st.data())  # doctest: +SKIP
    >>> draw_if_strategy(data.draw, 42)  # doctest: +SKIP
    42

    With a strategy, a value is drawn:

    >>> @st.composite  # doctest: +SKIP
    ... def demo(draw):
    ...     return draw_if_strategy(draw, st.just("hello"))
    >>> demo().example()  # doctest: +SKIP
    'hello'

    """
    return draw(v) if isinstance(v, st.SearchStrategy) else v
