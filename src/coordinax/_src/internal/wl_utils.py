"""Wadler-Lindig rendering utilities for coordinax internal types."""

__all__ = ("jax_scalar_handler", "pos_named_objs")

from collections.abc import Iterable
from typing import Any

import jax
import numpy as np
import wadler_lindig as wl


def jax_scalar_handler(obj: Any, /) -> wl.AbstractDoc | None:
    """Handler to render concrete 0-d JAX arrays as Python scalars.

    Pass this as ``custom=jax_scalar_handler`` to :func:`wadler_lindig.pdoc`,
    :func:`wadler_lindig.pformat`, or :func:`wadler_lindig.pprint` so that
    concrete 0-d JAX arrays are displayed as plain Python numbers (``10.0``)
    rather than the default array-summary form (``f64[](jax)``).

    Rendering rules:

    - **JAX Tracer** (inside ``jax.jit``): returns ``None`` so that
      ``wadler_lindig`` falls back to its default shape/dtype summary.
    - **Concrete 0-d array** (has ``.item()``): returns a doc for the
      plain Python scalar.
    - **Everything else**: returns ``None`` (default behaviour).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import wadler_lindig as wl
    >>> from coordinax.internal import jax_scalar_handler

    Concrete scalars inside a dict are shown as Python numbers:

    >>> d = {"x": jnp.array(10.0), "y": jnp.array(0.0)}
    >>> wl.pformat(d, custom=jax_scalar_handler)
    "{'x': 10.0, 'y': 0.0}"

    Callables are unaffected (``custom`` returns ``None``, wl uses default):

    >>> f = lambda t: {"x": t}
    >>> wl.pformat(f, custom=jax_scalar_handler)
    '<function <lambda>>'

    """  # noqa: D401
    # Tracers: let wadler_lindig render the shape/dtype summary.
    if isinstance(obj, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
        return None

    # Concrete 0-d arrays: convert to a Python scalar for display.
    if hasattr(obj, "item"):
        return wl.pdoc(obj.item())

    return None


def pos_named_objs(
    pairs: Iterable[tuple[str, Any]],
    pos_names: list[str],
    fields: dict[str, Any],
    /,
    *,
    hide_defaults: bool = True,
    **kw: Any,
) -> list[wl.AbstractDoc]:
    """Render positional fields first, then non-default named fields.

    Parameters
    ----------
    pairs
        Field name-value pairs (e.g., from ``field_items(self)``).
    pos_names
        Names of fields to render positionally (in order), listed first.
    fields
        Field descriptors (``self.__dataclass_fields__``). Used to read
        ``default`` values for filtering optional named fields.
    hide_defaults
        If ``True`` (default), named fields whose value equals the field's
        default are omitted from the output.
    **kw
        Extra keyword arguments forwarded to :func:`wadler_lindig.pdoc` and
        :func:`wadler_lindig.named_objs`.

    Returns
    -------
    list[wl.AbstractDoc]
        Positional docs followed by named non-default docs.

    """
    pairs_d = dict(pairs)
    pos_names_set = set(pos_names)
    kw["hide_defaults"] = hide_defaults
    pos_objs = [wl.pdoc(pairs_d[k], **kw) for k in pos_names if k in pairs_d]
    named_docs = wl.named_objs(
        [
            (k, v)
            for k, v in pairs_d.items()
            if k not in pos_names_set
            and (not hide_defaults or not np.array_equal(v, fields[k].default))
        ],
        **kw,
    )
    return [*pos_objs, *named_docs]
