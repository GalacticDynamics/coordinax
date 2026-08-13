"""Dispatch signatures coordinax owns must stay faithful to `plum`.

`plum` caches method resolution only while every method of a function is
*faithful* -- decidable from the argument types alone. One unfaithful method
turns the cache off for the whole function, and resolution then runs on every
call. The codebase already pays for this knowledge: `CDict` is a bare ``dict``
at runtime, not ``dict[CKey, Any]``, precisely so the signatures using it stay
faithful.

Nothing warns when a new annotation breaks it -- the code keeps working, only
slower -- so the check is here. It covers the methods coordinax registers; a
foreign package's unfaithful method on a shared name is not ours to fix.
"""

__all__: tuple[str, ...] = ()

import plum
import pytest

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import coordinax.vectors as cxv

#: Imported for the dispatch registrations they carry, not for their names.
_REGISTERED_BY = (cxc, cxf, cxm, cxr, cxfm, cxv)

#: Functions that dispatch on something types alone cannot decide, and the
#: reason. These are design choices, not accidents, and they cost the cache.
UNFAITHFUL_BY_DESIGN = {
    # Dispatches on trailing axis length via jaxtyping `Shaped[..., "*batch N"]`,
    # and on `frozenset[str]`; neither is decidable from the type.
    "guess_chart",
    # Dispatch on `type[...]`, i.e. on classes rather than instances.
    "guess_manifold",
    "dimension_of",
    # Dispatch on astropy `PhysicalType` unions.
    "guess_rep",
    "guess_basis_kind",
    "guess_geometry_kind",
    "guess_semantic_kind",
    # Dispatches on `Mapping[...]` keys and on a `Literal` sentinel.
    "uconvert",
}


def _coordinax_unfaithful() -> dict[str, list[str]]:
    """Core-`coordinax`-registered methods that are not faithful, by name.

    Restricted to the core package: whether a `coordinaxs.*` plugin has been
    imported depends on what else the session touched, and a plugin's
    signature is not this package's to fix.
    """
    out: dict[str, list[str]] = {}
    for name, f in plum.dispatch.functions.items():
        f._resolve_pending_registrations()
        for method in f.methods:
            module = getattr(method.implementation, "__module__", "") or ""
            if not module.startswith("coordinax."):
                continue
            if not method.signature.is_faithful:
                out.setdefault(name, []).append(f"{module}: {method.signature}")
    return out


def test_no_new_unfaithful_signatures() -> None:
    """A new unfaithful signature silently disables plum's method cache."""
    unexpected = {
        name: sigs
        for name, sigs in _coordinax_unfaithful().items()
        if name not in UNFAITHFUL_BY_DESIGN
    }
    assert not unexpected, (
        "These coordinax dispatch methods are not faithful, which turns off "
        "plum's method cache for the whole function -- resolution then runs on "
        "every call. Usually the cause is a parametric annotation such as "
        "`dict[str, Any]` where the bare container would do (see `CDict`). "
        f"{unexpected}"
    )


@pytest.mark.parametrize("name", sorted(UNFAITHFUL_BY_DESIGN))
def test_the_allowlist_has_no_stale_entries(name: str) -> None:
    """An entry that became faithful should be dropped, to keep the list honest."""
    assert name in _coordinax_unfaithful(), (
        f"`{name}` no longer has an unfaithful coordinax method; remove it "
        "from UNFAITHFUL_BY_DESIGN so a future regression is caught."
    )
