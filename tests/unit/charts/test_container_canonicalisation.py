"""A chart's declared dimensions decide its components' container types.

`coord_dimensions` says which components are angular; nothing used to make the
stored container agree. It tracked whichever arithmetic produced the value --
`Angle` survives a copy and degrades to `Quantity` through `Angle + Quantity`
or any trigonometric call -- so the same coordinate in the same chart came back
as `Angle` or `Quantity` depending on the route taken to reach it.

That is not cosmetic. `Angle` and `Quantity` are distinct pytree nodes, so a
route-dependent container is a route-dependent *pytree structure*: `jit` caches
miss, and `jax.tree.map` over two dicts of the same chart fails, according to
how each was obtained.
"""

import itertools
import linecache
from pathlib import Path

import jax
import pytest

import unxt as u

import coordinax._src.charts.register_ptmap
import coordinax._src.embedded.register_charts
import coordinax._src.product.chart
import coordinax._src.spherical.register_ptmap
import coordinax.charts as cxc
import coordinaxs.api.charts as cxcapi
from coordinax._src.charts.containers import canonical_containers
from coordinax._src.exceptions import NoGlobalCartesianChartError

#: The deliberate signal that a chart pair has no transition.
UNSUPPORTED = NoGlobalCartesianChartError

#: Every precondition guard in the `pt_map` rules, as it appears in source.
#: They are bare `assert`s, so an unsupported pair surfaces as a plain
#: `AssertionError` -- indistinguishable, by type, from a genuine broken
#: invariant anywhere else in the call. Matching the guard's own source line
#: keeps that distinction, so only these skip and everything else fails.
#: Enumerated from the transition modules rather than grown one failure at a
#: time; `test_guard_list_is_complete` fails if a new one appears.
_GUARDS = (
    "assert from_M ==",
    "assert to_M ==",
    "assert from_cart ==",
    "assert to_chart.M in",
    "assert to_M in",
    "assert from_chart.M in",
)


def _is_cross_manifold_guard(exc: AssertionError, /) -> bool:
    """Return whether `exc` came from one of the `pt_map` manifold guards."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return False
    line = linecache.getline(tb.tb_frame.f_code.co_filename, tb.tb_lineno).strip()
    return line.startswith(_GUARDS)


#: One value per component name, in the container the chart declares.
SAMPLE = {
    "x": u.Q(1.0, "m"),
    "y": u.Q(2.0, "m"),
    "z": u.Q(3.0, "m"),
    "r": u.Q(3.0, "m"),
    "rho": u.Q(2.0, "m"),
    "distance": u.Q(3.0, "m"),
    "theta": u.Angle(1.0, "rad"),
    "phi": u.Angle(0.5, "rad"),
    "lon": u.Angle(0.5, "rad"),
    "lat": u.Angle(0.3, "rad"),
    "lon_coslat": u.Angle(0.4, "rad"),
    "t": u.Q(1.0, "s"),
}

CHARTS = [
    cxc.cart1d,
    cxc.radial1d,
    cxc.cart2d,
    cxc.polar2d,
    cxc.cart3d,
    cxc.cyl3d,
    cxc.sph3d,
    cxc.math_sph3d,
    cxc.lonlat_sph3d,
    cxc.loncoslat_sph3d,
    cxc.sph2,
    cxc.lonlat_sph2,
    cxc.math_sph2,
    cxc.loncoslat_sph2,
    cxc.sph1,
]

PAIRS = list(itertools.product(CHARTS, CHARTS))


def _point(chart):
    """Return a sample point for `chart`, or None if a component has no sample."""
    try:
        return {k: SAMPLE[k] for k in chart.components}
    except KeyError:
        return None


class TestPtMapCanonicalisesContainers:
    """Every chart transition lands its angles in `Angle`, by either entry form."""

    @pytest.mark.parametrize(("frm", "to"), PAIRS)
    def test_three_argument_form(self, frm, to) -> None:
        p = _point(frm)
        if p is None:
            pytest.skip("no sample for this chart's components")
        try:
            out = cxcapi.pt_map(p, frm, to)
        except UNSUPPORTED:
            pytest.skip("transition not implemented")
        except AssertionError as exc:
            if not _is_cross_manifold_guard(exc):
                raise
            pytest.skip("cross-manifold pair; no transition")
        assert isinstance(out, dict)
        # Canonicalising an already-canonical point returns it unchanged.
        assert canonical_containers(out, to) is out

    @pytest.mark.parametrize(("frm", "to"), PAIRS)
    def test_five_argument_form(self, frm, to) -> None:
        """The explicit-manifold form must agree.

        It dispatches straight to the specific pair, bypassing the
        three-argument entry.
        """
        p = _point(frm)
        if p is None:
            pytest.skip("no sample for this chart's components")
        try:
            out = cxcapi.pt_map(p, frm.M, frm, to.M, to)
        except UNSUPPORTED:
            pytest.skip("transition not implemented")
        except AssertionError as exc:
            if not _is_cross_manifold_guard(exc):
                raise
            pytest.skip("cross-manifold pair; no transition")
        assert isinstance(out, dict)
        # Canonicalising an already-canonical point returns it unchanged.
        assert canonical_containers(out, to) is out


class TestStructureIsRouteIndependent:
    """The point of canonicalising: pytree structure must not depend on route."""

    def test_same_chart_same_treedef_by_any_route(self) -> None:
        direct = cxcapi.pt_map(_point(cxc.cart3d), cxc.cart3d, cxc.sph3d)
        viacyl = cxcapi.pt_map(
            cxcapi.pt_map(_point(cxc.cart3d), cxc.cart3d, cxc.cyl3d),
            cxc.cyl3d,
            cxc.sph3d,
        )
        assert jax.tree.structure(direct) == jax.tree.structure(viacyl)

    def test_tree_map_against_a_transition_result(self) -> None:
        """`tree.map` over a hand-built point and a returned one must not raise."""
        built = _point(cxc.sph3d)
        returned = cxcapi.pt_map(_point(cxc.cart3d), cxc.cart3d, cxc.sph3d)
        jax.tree.map(lambda a, b: a, built, returned)


class TestIdentityIsPreserved:
    """Canonicalising must not cost the no-op transition its identity."""

    @pytest.mark.parametrize("chart", [cxc.sph2, cxc.cart3d])
    def test_canonical_input_returns_the_same_object(self, chart) -> None:
        p = _point(chart)
        assert cxcapi.pt_map(p, chart, chart) is p

    def test_non_canonical_input_is_canonicalised(self) -> None:
        """A new dict here is correct.

        The alternative is the identity route preserving a container that every
        other route would have normalised.
        """
        p = {"theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
        out = cxcapi.pt_map(p, cxc.sph2, cxc.sph2)
        assert out is not p
        assert canonical_containers(out, cxc.sph2) is out


def test_guard_list_is_complete() -> None:
    """`_GUARDS` must name every `assert` in the transition modules.

    The list decides what this file is willing to skip. If a rule grows a new
    precondition and it is not here, its failures become skips -- silently, and
    exactly the way the containers being canonicalised went wrong in the first
    place.
    """
    modules = [
        Path(m.__file__)
        for m in (
            coordinax._src.charts.register_ptmap,
            coordinax._src.spherical.register_ptmap,
            coordinax._src.product.chart,
            coordinax._src.embedded.register_charts,
        )
    ]
    found = {
        line.strip().split("  #")[0].strip()
        for path in modules
        for line in path.read_text().splitlines()
        if line.strip().startswith("assert ")
    }
    unlisted = {a for a in found if not a.startswith(_GUARDS)}
    assert not unlisted, f"unlisted assert guards: {sorted(unlisted)}"
