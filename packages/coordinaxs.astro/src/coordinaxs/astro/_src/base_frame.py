"""Astronomy reference frames."""

__all__ = ("AbstractSpaceFrame",)


from coordinax.frames import AbstractReferenceFrame


class AbstractSpaceFrame(AbstractReferenceFrame, is_abstract=True):
    r"""Abstract base class for astronomy-oriented spatial reference frames.

    This class specializes {class}`coordinax.frames.AbstractReferenceFrame` for
    frames that act on **spatial** coordinates (for example,
    {class}`coordinaxs.astro.ICRS` and {class}`coordinaxs.astro.Galactocentric`).
    In the terminology of the coordinax specification, frame changes are
    interpreted as **active** transformations: operators act directly on points
    and move them on the same manifold.

    Conceptually, a frame transition corresponds to a smooth map $F : M \to
    M$ on the same spatial manifold, with chart-level formulas supplied by
    registered ``frame_transition`` dispatches.

    Notes
    -----
    - ``AbstractSpaceFrame`` is a typing and dispatch category; concrete frames
        should subclass it and define the parameters that characterize that
        frame.
    - Transform operators are produced via
        {func}`coordinax.frames.frame_transition`.
    - The generic astronomy-space fallback composes transformations through
        ICRS, so a custom spatial frame must register **both** ICRS legs --
        ``(MyFrame, ICRS)`` and ``(ICRS, MyFrame)``. One direction alone leaves
        the other, and the frame's self-transition, unroutable; both then raise
        {class}`coordinax.frames.FrameTransformError`.
    - This class is for 3D spatial frame semantics; spacetime coordinate models
        (for example, where ``ct`` is part of the point itself) are represented
        separately from this spatial frame category.

    Examples
    --------
    >>> import plum
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> import coordinaxs.astro as cxastro

    >>> class MySpaceFrame(cxastro.AbstractSpaceFrame):
    ...     pass

    >>> @plum.dispatch
    ... def frame_transition(
    ...     from_frame: MySpaceFrame, to_frame: cxastro.ICRS, /
    ... ) -> cxfm.Identity:
    ...     return cxfm.identity

    >>> @plum.dispatch
    ... def frame_transition(
    ...     from_frame: cxastro.ICRS, to_frame: MySpaceFrame, /
    ... ) -> cxfm.Identity:
    ...     return cxfm.identity

    >>> op = cxf.frame_transition(MySpaceFrame(), cxastro.ICRS())
    >>> op
    Identity()

    Registering neither leg -- or only one of them -- is an error rather than a
    runaway descent through the ICRS fallback:

    >>> class Unrouted(cxastro.AbstractSpaceFrame):
    ...     pass

    >>> try:
    ...     cxf.frame_transition(Unrouted(), cxastro.galactic)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    No `frame_transition` registered from Unrouted to ICRS.

    The class is a dispatch category, not a frame, so it cannot be built:

    >>> try:
    ...     cxastro.AbstractSpaceFrame()
    ... except TypeError as e:
    ...     print(e)
    Cannot instantiate abstract `equinox.Module`.

    """
