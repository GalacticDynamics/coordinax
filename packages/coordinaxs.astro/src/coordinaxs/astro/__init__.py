"""``import coordinaxs.astro as cxastro`` — Frames for Astronomy."""

__all__ = (
    "Parallax",
    "DistanceModulus",
    "AbstractSpaceFrame",
    "ICRS",
    "icrs",
    "Galactic",
    "galactic",
    "Galactocentric",
)

from ._setup_package import install_import_hook

# INVARIANT: this block binds all of astro's public symbols and MUST run before
# the frames/interop re-invocation below. Those hooks complete registration that
# was deferred while astro was importing (see the note there); they can only
# succeed once these names exist. Reordering the re-invocation above this block
# silently reintroduces the "astropy conversions never registered" bug for the
# astro-first import ordering.
with install_import_hook("coordinaxs.astro"):
    from ._src import (
        ICRS,
        AbstractSpaceFrame,
        DistanceModulus,
        Galactic,
        Galactocentric,
        Parallax,
        galactic,
        icrs,
    )


# Now that astro's symbols exist, (a) populate optional exports into
# `coordinax.frames`, and (b) complete any interop registration that was
# deferred while this package was initializing: when `coordinaxs.astro` is
# imported before `coordinax`, core's interop loader runs mid-import here and
# leaves the astropy entry point pending because astro's types are not yet
# resolvable, so retrying now registers the conversions.
#
# `coordinax` may itself still be initializing (it pulls astro in through the
# `coordinaxs.frames` entry point while executing `coordinax.frames`), in which
# case the interop loader is not defined yet — core runs it at the end of its
# own import, so there is nothing to retry here.
import coordinax as cx
import coordinax.frames as cxf

cxf._load_optional_frame_exports()

_load_interop = getattr(cx, "_load_optional_interop", None)
if _load_interop is not None:
    _load_interop()

del cxf, cx, _load_interop, install_import_hook
