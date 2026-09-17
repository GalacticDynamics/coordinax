r"""The API report's dispatch filter must exclude dispatch and nothing else.

`scripts/api_report.py` hides `plum.Function` names, because griffe reads
source statically and so sees only the last of a verb's registrations. The
filter is the whole safety of that arrangement: exclude too little and the
report fills with fabricated breaks; exclude too much and it drops real ones
silently, which is worse, because an empty report reads as "nothing changed".

The first version matched *bare* names against every segment of a dotted path.
Eight modules in this package are named after a dispatched verb -- `norm.py`,
`interval.py`, `add.py`, `angle_between.py`, `chord_distance.py`,
`geodesic_distance.py`, `scale_factors.py`, `tangent_map.py` -- so every object
defined in any of them was excluded, dispatch or not, along with methods
sharing a verb's name. These pin the collision cases, which are invisible
unless looked for. See #924.
"""

__all__: tuple[str, ...] = ()

import importlib.util
import pathlib
import sys

import pytest

# Loaded by path, not imported. `scripts/` is not part of the installed
# distribution, and the suite runs under `--import-mode=importlib`, which
# deliberately leaves the rootdir off `sys.path` -- so `import scripts` works
# under `python -m pytest` (which adds the cwd) and fails under the `pytest`
# console script that nox and CI use. Loading the file directly works under
# both, and does not put the repo root on the path for every other test.
_SPEC = importlib.util.spec_from_file_location(
    "_api_report_under_test",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "api_report.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
_api_report = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _api_report
_SPEC.loader.exec_module(_api_report)

_is_dispatched = _api_report._is_dispatched
dispatched_paths = _api_report.dispatched_paths

#: A dispatched verb's own qualified path, and a same-named module beside it.
_EXCLUDED = frozenset(
    {
        "coordinax._src.charts.register_ptmap.pt_map",
        "coordinax._src.manifolds.norm.norm",
    }
)


@pytest.mark.parametrize(
    "path",
    [
        "coordinax._src.charts.register_ptmap.pt_map",
        "coordinax._src.manifolds.norm.norm",
    ],
)
def test_a_dispatched_verb_is_excluded(path: str) -> None:
    """The case the filter exists for."""
    assert _is_dispatched(path, _EXCLUDED)


@pytest.mark.parametrize(
    "path",
    [
        # a class living in the module *named* after a verb
        "coordinax._src.manifolds.norm.SomeClass",
        # ... and one of its methods
        "coordinax._src.manifolds.norm.SomeClass.method",
        # a method whose own name is a verb, on an unrelated class
        "coordinax.Coordinate.cconvert",
        # a class in another same-named module
        "coordinax._src.interval.Interval",
        # a verb's name as a bare prefix of a longer name
        "coordinax._src.manifolds.norm_helpers.thing",
    ],
)
def test_a_name_collision_is_not_excluded(path: str) -> None:
    """What the bare-name filter dropped, silently."""
    assert not _is_dispatched(path, _EXCLUDED)


def test_something_inside_a_dispatched_function_is_excluded() -> None:
    """A descendant travels with its parent, so a nested object is dropped too."""
    assert _is_dispatched(
        "coordinax._src.charts.register_ptmap.pt_map.inner", _EXCLUDED
    )


def test_the_live_package_still_has_a_dispatched_surface_to_exclude() -> None:
    """If `pt_map` stops being dispatched, the filter is excluding nothing.

    The script raises on this rather than report noise; this catches it here,
    where the message can say why, instead of in CI.
    """
    paths = dispatched_paths()
    assert any(p.endswith(".pt_map") for p in paths), (
        "no `pt_map` among the dispatched paths -- either dispatch was removed "
        "or `dispatched_paths` stopped finding it; the report would be noise"
    )
