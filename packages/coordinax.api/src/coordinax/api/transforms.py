"""Representations."""

__all__: tuple[str, ...] = ("act", "compose")

from typing import Any

import plum


@plum.dispatch.abstract
def act(*args: Any, **kwargs: Any) -> Any:
    r"""Apply a transform action to coordinates.

    This is the core dispatch function for transform application. Each transform
    type registers its own implementation via multiple dispatch. Transforms act
    on various input types (Array, Quantity, Vector, CDict) according to their
    semantics.

    Mathematical Definition:

    For a transform $\mathcal{T}$ parameterized by $\tau$, this computes:

    $$ x' = \mathcal{T}(\tau)(x) $$

    For tau-independent transforms, $\tau$ is ignored. For composite transforms
    (e.g., ``Composed``), the component transforms are applied sequentially.

    Parameters
    ----------
    op : coordinax.transforms.AbstractTransform
        The transform to apply. This can be any transform type:

        - ``Translate``: Spatial translation (point geometry)
        - ``Rotate``: Spatial rotation
        - ``Identity``: No-op
        - ``Composed``: Sequential composition

    tau : Any
        Parameter for tau-dependent transforms. Pass ``None`` for
        tau-independent transforms.

    x : Any
        The input to transform. Supported types depend on the transform:

        - ``Array``/``ArrayLike``: Interpreted as Cartesian point data
        - ``Quantity``: Unitful array, treated as Cartesian point
        - ``Vector``: Role-aware transformation with chart preservation
        - ``CDict``: Low-level component dict

    *args, **kwargs : Any
        Additional positional/keyword arguments passed to concrete dispatches,
        e.g. ``chart``, ``rep``, ``usys``.

    Returns
    -------
    Any
        The transformed input, same type as ``x``.

    Raises
    ------
    NotImplementedError
        If no dispatch is registered for the given ``(transform, input)`` types.

    Notes
    -----
    - **Transform.__call__**: The ``__call__`` method of transforms delegates
      to this function: ``op(tau, x)`` is equivalent to ``act(op, tau, x)``.

    - **Chart inference**: When no chart is provided and the input is an
      Array or Quantity, the chart is inferred via ``coordinax.charts.guess_chart``.

    - **Composite transforms**: For ``Composed``, the component transforms are
      applied in sequence (left-to-right).

    See Also
    --------
    coordinax.transforms.act : Concrete dispatch entrypoint used in practice
    coordinax.frames.compose : Compose two transforms into one
    coordinax.frames.simplify : Simplify a transform to canonical form

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    **Apply a rotation to a Quantity:**

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxfm.act(op, None, q).round(3)
    Q([0., 1., 0.], 'km')

    **Apply a translation to a Quantity (usys required):**

    >>> import jax.numpy as jnp
    >>> op = cxfm.Translate.from_([1, 0, 0], "km")
    >>> x = jnp.asarray([1.0, 0.0, 0.0])  # metres (dimensionless array)
    >>> cxfm.act(op, None, x, usys=u.unitsystems.si).round(3)
    Array([1001.,    0.,    0.], dtype=float64)

    **Composite transform:**

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> T = cxfm.Translate.from_([1, 0, 0], "km")
    >>> op = R | T  # rotate then translate
    >>> cxfm.act(op, None, q).round(3)
    Q([1., 1., 0.], 'km')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def compose(*args: Any, **kwargs: Any) -> Any:
    """Compose two frame transforms into a single transform.

    Examples
    --------
    >>> import coordinax.transforms as cxfm
    >>> import unxt as u

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    >>> cxfm.compose(shift, rotate)
    Composed(( Translate(...), Rotate(...) ))

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def simplify(*args: Any, **kwargs: Any) -> Any:
    """Simplify a transform to a canonical form.

    This function takes a transform and attempts to simplify it, returning a
    new, potentially simpler transform. For example, a ``Translate`` with zero
    delta simplifies to ``Identity``.

    Notes
    -----
    In general this cannot be called in a JIT'ed context because it generally
    requires inspecting values to determine if simplifications are possible.

    This function uses multiple dispatch. Each operator type registers its own
    simplification rules.

    To see all available dispatches::

        >>> import coordinax.transforms as cxfm
        >>> cxfm.simplify.methods  # doctest: +SKIP
        List of 7 method(s):
            [0] simplify(...)

    Examples
    --------
    >>> import coordinax.transforms as cxfm

    **Identity (already simple):**

    >>> op = cxfm.Identity()
    >>> cxfm.simplify(op) is op
    True

    **Translate with zero delta:**

    >>> op = cxfm.Translate.from_([0, 0, 0], "m")
    >>> cxfm.simplify(op)
    Identity()

    **Translate with non-zero delta (no simplification):**

    >>> op = cxfm.Translate.from_([1, 2, 3], "m")
    >>> simplified = cxfm.simplify(op)
    >>> type(simplified).__name__
    'Translate'

    **Rotate with identity matrix:**

    >>> import unxt as u
    >>> op = cxfm.Rotate.from_euler("z", u.Q(0, "deg"))
    >>> cxfm.simplify(op)
    Identity()

    """
    raise NotImplementedError  # pragma: no cover
