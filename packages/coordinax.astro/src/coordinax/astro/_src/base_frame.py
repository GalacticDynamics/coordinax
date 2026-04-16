"""Astronomy reference frames."""

__all__ = ("AbstractSpaceFrame",)


from coordinax.frames import AbstractReferenceFrame


class AbstractSpaceFrame(AbstractReferenceFrame):
    r"""Abstract base class for astronomy-oriented spatial reference frames.

    This class specializes {class}`coordinax.frames.AbstractReferenceFrame` for
    frames that act on **spatial** coordinates (for example,
    {class}`coordinax.astro.ICRS` and {class}`coordinax.astro.Galactocentric`).
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
        ICRS, so custom spatial frames should typically register at least one
        transformation path to or from ICRS.
    - This class is for 3D spatial frame semantics; spacetime coordinate models
        (for example, where ``ct`` is part of the point itself) are represented
        separately from this spatial frame category.

    Examples
    --------
    >>> import plum
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.astro as cxastro

    >>> class MySpaceFrame(cxastro.AbstractSpaceFrame):
    ...     pass

    >>> @plum.dispatch
    ... def frame_transition(
    ...     from_frame: MySpaceFrame, to_frame: cxastro.ICRS, /
    ... ) -> cxfm.Identity:
    ...     return cxfm.identity

    >>> op = cxf.frame_transition(MySpaceFrame(), cxastro.ICRS())
    >>> op
    Identity()

    """
