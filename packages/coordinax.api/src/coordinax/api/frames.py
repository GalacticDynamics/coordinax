"""Representations."""

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
    - This function uses multiple dispatch. To inspect all registered pairs::

        >>> import coordinax.frames as cxf
        >>> cxf.frame_transition.methods  # doctest: +SKIP
        List of 20 method(s):
            [0] frame_transition(from_frame: ...)

    See Also
    --------
    coordinax.frames.act : Apply a transform to coordinates
    coordinax.frames.compose : Compose two transforms into one

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
    >>> from coordinax.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(ICRS(), R)

    >>> op = cxf.frame_transition(ICRS(), frame)
    >>> type(op).__name__
    'Composed'

    """
    raise NotImplementedError  # pragma: no cover
