"""Frame API for coordinax: the transition between reference frames."""

__all__: tuple[str, ...] = ("frame_transition",)

from typing import Any

import plum


@plum.dispatch.abstract
def frame_transition(*args: Any, **kwargs: Any) -> Any:
    """Return the transform operator that maps coordinates from one frame to another.

    Given a source frame and a target frame, ``frame_transition`` computes the
    ``AbstractTransform`` (or composed chain of transforms) that, when applied
    to coordinates expressed in *from_frame*, yields coordinates expressed in
    *to_frame*.

    Notes
    -----
    - Each pair of concrete frame types registers its own dispatch.
      Calling ``frame_transition(frame_a, frame_a)`` returns ``Identity()``.
    - For ``TransformedReferenceFrame``, the transition is constructed
      automatically by composing the base-frame transition with the stored
      ``xop``.
    - The returned transform is invertible: ``op.inverse`` gives the
      *to_frame* → *from_frame* direction.
    - **Build the operator outside ``jit``, then pass it in as an argument.**
      ``frame_transition`` is pure Python: it walks the dispatch table,
      composes the chain and simplifies it. That work does not belong inside a
      traced function -- do it once, up front, and hand the operator to the
      jitted function that applies it. Frames and transforms are pytrees, so
      changing frame *parameters* reuses the same compiled code; closing over
      the operator instead makes it static and buys a fresh compile per
      operator. The payoff is large: applying a precomputed operator under
      ``jit`` runs in microseconds rather than milliseconds (roughly 30x at
      ``N=1``, and over 100x against ``to_frame``, which rebuilds the operator
      on every call). Eager application costs a few milliseconds of fixed
      Python plus tens of nanoseconds per element, so below ~1e5 elements you
      are paying almost entirely for Python.
    - This function uses multiple dispatch. To inspect all registered pairs::

        >>> import coordinax.frames as cxf
        >>> cxf.frame_transition.methods  # doctest: +SKIP
        List of 20 method(s):
            [0] frame_transition(from_frame: ...)

    See Also
    --------
    coordinax.transforms.act : Apply a transform to coordinates
    coordinax.transforms.compose : Compose two transforms into one

    Examples
    --------
    >>> import coordinax.frames as cxf

    **Same-to-same frame (identity):**

    >>> cxf.frame_transition(cxf.alice, cxf.alice)
    Identity()

    **Alice → Alex:**

    >>> op = cxf.frame_transition(cxf.alice, cxf.alex)
    >>> op
    Composed(( Translate(...), Rotate(...) ))

    **Alex → Alice (inverse direction):**

    >>> op = cxf.frame_transition(cxf.alex, cxf.alice)
    >>> op
    Composed(( Rotate(...), Translate(...) ))

    **Using a TransformedReferenceFrame:**

    >>> import coordinax.transforms as cxfm
    >>> import quaxed.numpy as jnp
    >>> from coordinaxs.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(ICRS(), R)

    >>> op = cxf.frame_transition(ICRS(), frame)
    >>> type(op).__name__
    'Composed'

    **Under ``jit`` -- operator built outside, passed in:**

    >>> import equinox as eqx
    >>> import coordinax as cx
    >>> import coordinax.transforms as cxfm

    >>> op = cxf.frame_transition(cxf.alice, cxf.alex)

    >>> @eqx.filter_jit
    ... def to_alex(op, p):
    ...     return cxfm.act(op, None, p)

    >>> p = cx.Point.from_([1, 2, 3], "kpc", cxf.alice)
    >>> print(to_alex(op, p))
    <Point: chart=Cart3D (x, y, z) [kpc]
        [-2.  1.  3.]>

    """
    raise NotImplementedError  # pragma: no cover
