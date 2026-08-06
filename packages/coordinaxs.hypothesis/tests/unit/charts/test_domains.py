"""Chart coordinate domains, and that `cdicts` actually respects them.

The domains are the executable form of a claim `coord_dimensions` cannot make:
which *values* a chart's components may take. These tests are the spec -- if a
chart's domain is wrong, or `cdicts` stops honouring one, they fail here rather
than as a mysterious singularity in whatever downstream test drew the point.
"""

__all__: tuple[str, ...] = ()

import math

import hypothesis.strategies as st
import pytest
import unxt as u
from hypothesis import HealthCheck, given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc

import coordinaxs.hypothesis.main as cxst
from coordinaxs.hypothesis.charts import Interval, component_domains
from coordinaxs.hypothesis.utils import get_all_subclasses

#: Charts with a module-level singleton, which is every chart that needs no
#: construction arguments.
CHARTS = [
    inst
    for cls in sorted(
        get_all_subclasses(cxc.AbstractChart, exclude_abstract=True),
        key=lambda c: c.__name__,
    )
    if isinstance(inst := getattr(cxc, cls.__name__.lower(), None), cxc.AbstractChart)
]
CHART_IDS = [type(c).__name__ for c in CHARTS]


def _in(interval: Interval, q: u.AbstractQuantity) -> bool:
    """Whether *q* lies inside *interval*, compared in the interval's unit."""
    if interval.unit is None:
        return True
    v = float(u.ustrip(interval.unit, q))
    lo = interval.min
    hi = interval.max
    if lo is not None and (v < lo or (interval.exclude_min and v <= lo)):
        return False
    return not (hi is not None and (v > hi or (interval.exclude_max and v >= hi)))


class TestDomainsAreWellFormed:
    """Every chart reports a domain for exactly its own components."""

    @pytest.mark.parametrize("chart", CHARTS, ids=CHART_IDS)
    def test_keys_match_components(self, chart: cxc.AbstractChart) -> None:
        assert set(component_domains(chart)) == set(chart.components)

    @pytest.mark.parametrize("chart", CHARTS, ids=CHART_IDS)
    def test_bounds_are_ordered(self, chart: cxc.AbstractChart) -> None:
        for name, interval in component_domains(chart).items():
            if interval.min is not None and interval.max is not None:
                assert interval.min < interval.max, name

    @pytest.mark.parametrize("chart", CHARTS, ids=CHART_IDS)
    def test_margin_leaves_room(self, chart: cxc.AbstractChart) -> None:
        """A margin must not eat the whole interval."""
        for name, interval in component_domains(chart).items():
            lo, hi = interval.bounds_in(u.unit(interval.unit or ""))
            if lo is not None and hi is not None:
                assert lo < hi, f"{name}: margin closed the interval"


class TestConventionsAreDistinguished:
    """The case that forces dispatch: same names, same dimensions, opposite domains."""

    def test_spherical_pair_is_indistinguishable_by_name_or_dimension(self) -> None:
        """Establishes *why* a name- or dimension-keyed lookup cannot work."""
        assert cxc.sph3d.components == cxc.math_sph3d.components
        assert cxc.sph3d.coord_dimensions == cxc.math_sph3d.coord_dimensions

    def test_physics_convention_theta_is_the_colatitude(self) -> None:
        theta = component_domains(cxc.sph3d)["theta"]
        assert theta.min == 0.0
        assert theta.max == pytest.approx(math.pi)

    def test_maths_convention_theta_is_the_azimuth(self) -> None:
        theta = component_domains(cxc.math_sph3d)["theta"]
        assert theta.min == pytest.approx(-math.pi)
        assert theta.max == pytest.approx(math.pi)

    def test_the_two_conventions_disagree(self) -> None:
        assert component_domains(cxc.sph3d) != component_domains(cxc.math_sph3d)

    def test_polar2d_theta_is_an_azimuth_not_a_colatitude(self) -> None:
        """`Polar2D` shares the name `theta` with `Spherical3D`'s colatitude."""
        assert component_domains(cxc.polar2d)["theta"].min == pytest.approx(-math.pi)


class TestBoundsFollowTheUnit:
    """Bounds are stated in one unit and must hold in whatever unit is drawn."""

    @pytest.mark.parametrize(
        ("unit_str", "expected_hi"),
        [("rad", math.pi), ("deg", 180.0), ("cycle", 0.5), ("arcmin", 180 * 60)],
    )
    def test_polar_upper_bound_converts(
        self, unit_str: str, expected_hi: float
    ) -> None:
        _, hi = component_domains(cxc.sph3d)["theta"].bounds_in(u.unit(unit_str))
        # The margin pulls it in slightly; the scale is what matters here.
        assert hi == pytest.approx(expected_hi, rel=0.05)

    def test_unconstrained_component_has_no_bounds(self) -> None:
        lo, hi = component_domains(cxc.cart3d)["x"].bounds_in(u.unit("m"))
        assert lo is None
        assert hi is None


class TestCDictsRespectsDomains:
    """The payoff: generated points are inside the domain, with no filtering."""

    @pytest.mark.parametrize("chart", CHARTS, ids=CHART_IDS)
    def test_draws_are_in_domain(self, chart: cxc.AbstractChart) -> None:
        domains = component_domains(chart)

        @given(point=cxst.cdicts(chart))
        @settings(
            max_examples=50, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            for name, q in point.items():
                assert _in(domains[name], q), f"{name}={q!r} outside {domains[name]}"

        check()

    def test_spherical_radius_is_positive(self) -> None:
        """Was 60% negative before the domains existed."""

        @given(point=cxst.cdicts(cxc.sph3d))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            assert float(u.ustrip("m", point["r"])) > 0

        check()

    def test_colatitude_is_off_both_poles(self) -> None:
        """Was 94% outside (0, pi) before the domains existed."""

        @given(point=cxst.cdicts(cxc.sph3d))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            theta = float(u.ustrip("rad", point["theta"]))
            assert 0 < theta < math.pi

        check()

    def test_units_still_vary(self) -> None:
        """Constraining values must not have collapsed unit diversity.

        Drawing a variety of units is the reason `cdicts` is useful for
        exercising unit handling; the domains convert bounds into the drawn
        unit rather than pinning it, so this must survive.
        """
        seen: set[tuple[str, ...]] = set()

        @given(point=cxst.cdicts(cxc.sph3d))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            seen.add(tuple(str(q.unit) for q in point.values()))

        collect()
        assert len(seen) > 5


class TestMagnitude:
    """The cap that makes draws usable in numerical comparisons."""

    def test_default_bounds_magnitude(self) -> None:
        @given(point=cxst.cdicts(cxc.cart3d))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            for q in point.values():
                assert abs(float(u.ustrip("m", q))) <= 1e3 * (1 + 1e-6)

        check()

    def test_explicit_magnitude_is_honoured(self) -> None:
        @given(point=cxst.cdicts(cxc.cart3d, magnitude=1.0))
        @settings(
            max_examples=100, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            for q in point.values():
                assert abs(float(u.ustrip("m", q))) <= 1.0 * (1 + 1e-6)

        check()

    def test_none_opts_out(self) -> None:
        """`magnitude=None` restores the unbounded behaviour, for stress tests."""
        biggest = 0.0

        @given(point=cxst.cdicts(cxc.cart3d, magnitude=None))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            nonlocal biggest
            for q in point.values():
                biggest = max(biggest, abs(float(u.ustrip("m", q))))

        collect()
        assert biggest > 1e3


class TestMagnitudeFloor:
    """A `(floor, cap)` range also keeps radial coordinates off the origin.

    The cap alone is not enough for a test that compares derivatives: a
    Jacobian entry scaling like ``1/r`` is unusable at ``r = 1e-3`` however
    modest the upper bound. This is what lets `cdicts` replace a hand-rolled
    per-component strategy.
    """

    def test_floor_applies_to_a_radius(self) -> None:
        radii = []

        @given(point=cxst.cdicts(cxc.sph3d, magnitude=(0.5, 8.0)))
        @settings(
            max_examples=150, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            radii.append(float(u.ustrip("m", point["r"])))

        collect()
        assert min(radii) >= 0.5 * (1 - 1e-6)
        assert max(radii) <= 8.0 * (1 + 1e-6)

    def test_floor_does_not_apply_to_a_free_axis(self) -> None:
        """``x = 0`` is an ordinary Cartesian value, not a degeneracy.

        Applying the floor everywhere would carve a hole out of the middle of
        every free axis.
        """
        seen_small = False

        @given(point=cxst.cdicts(cxc.cart3d, magnitude=(0.5, 8.0)))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            nonlocal seen_small
            if abs(float(u.ustrip("m", point["x"]))) < 0.5:
                seen_small = True

        collect()
        assert seen_small

    def test_scalar_magnitude_still_means_a_cap(self) -> None:
        """Backwards-compatible: a bare number is the upper bound only."""

        @given(point=cxst.cdicts(cxc.sph3d, magnitude=8.0))
        @settings(
            max_examples=100, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            assert float(u.ustrip("m", point["r"])) <= 8.0 * (1 + 1e-6)

        check()


class TestElementsInteraction:
    """A caller-supplied `elements` is honoured but still held to the domain."""

    def test_elements_narrows_within_the_domain(self) -> None:
        @given(
            point=cxst.cdicts(
                cxc.sph3d, elements=st.floats(0.5, 2.0, width=32), magnitude=None
            )
        )
        @settings(
            max_examples=100, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            assert float(u.ustrip("m", point["r"])) > 0
            assert 0 < float(u.ustrip("rad", point["theta"])) < math.pi

        check()

    def test_elements_cannot_reintroduce_invalid_coordinates(self) -> None:
        """An `elements` range outside the domain fails loudly, not silently.

        Filtering rather than overriding is the deliberate choice here: a
        caller who asks for negative radii gets `Unsatisfiable`, instead of a
        `Spherical3D` point with r < 0 that only misbehaves much later.
        """

        @given(point=cxst.cdicts(cxc.sph3d, elements=st.floats(-10.0, -1.0, width=32)))
        @settings(
            max_examples=25, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:  # pragma: no cover - never reached
            pytest.fail("a negative radius should not be reachable")

        with pytest.raises(Unsatisfiable):
            check()
