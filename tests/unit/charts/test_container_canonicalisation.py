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

import jax
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinaxs.api.charts as cxcapi
from coordinax._src.charts.containers import canonical_containers
from coordinax._src.exceptions import NoGlobalCartesianChartError

#: Unsupported pairs signal via these two, and nothing else does. Catching more
#: broadly would let a genuine regression register as a skip.
UNSUPPORTED = (NoGlobalCartesianChartError, AssertionError)

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
