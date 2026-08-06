"""Chart coordinate domains, and that `cdicts` actually respects them.

The domains are the executable form of a claim `coord_dimensions` cannot make:
which *values* a chart's components may take. These tests are the spec -- if a
chart's domain is wrong, or `cdicts` stops honouring one, they fail here rather
than as a mysterious singularity in whatever downstream test drew the point.
"""

__all__: tuple[str, ...] = ()

import math

from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import unxt as u
from hypothesis import HealthCheck, given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
import coordinax.manifolds as cxm

import coordinaxs.hypothesis.main as cxst
from coordinaxs.hypothesis.charts import Interval, component_domains
from coordinaxs.hypothesis.utils import get_all_subclasses

#: Every concrete chart, one instance each.
#:
#: Enumerated from the module's own singletons rather than guessed from class
#: names: the singleton for `Spherical3D` is `sph3d`, not `spherical3d`, so a
#: ``getattr(cxc, cls.__name__.lower())`` heuristic silently collects only the
#: Cartesian-ish half and skips every chart with an interesting domain.
_SINGLETONS = sorted(
    {
        id(obj): obj for obj in vars(cxc).values() if isinstance(obj, cxc.AbstractChart)
    }.values(),
    key=lambda c: type(c).__name__,
)

#: The three that take construction arguments, so have no singleton.
_CONSTRUCTED = [
    cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1.0, "kpc")),
    cxc.CartesianProductChart(
        factors=(cxc.cart3d, cxc.polar2d), factor_names=("q", "p")
    ),
    cxm.EmbeddedChart(embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "kpc"))),
]

CHARTS = [*_SINGLETONS, *_CONSTRUCTED]
CHART_IDS = [type(c).__name__ for c in CHARTS]


def test_every_concrete_chart_is_covered() -> None:
    """The list above must not drift behind the chart hierarchy.

    Without this, adding a chart silently adds an untested domain -- which is
    exactly how the original name-guessing version came to skip 13 of 24.
    """
    concrete = {
        c.__name__ for c in get_all_subclasses(cxc.AbstractChart, exclude_abstract=True)
    }
    assert concrete - {type(c).__name__ for c in CHARTS} == set()


def _in(interval: Interval, q: u.AbstractQuantity) -> bool:
    """Whether *q* lies inside *interval*, margins included.

    Compares in the interval's own unit, and against the margin-adjusted
    bounds -- a draw sitting exactly on an open bound is outside.
    """
    if interval.unit is None:
        return True
    lo, hi = interval.bounds_in(u.unit(interval.unit))
    v = float(u.ustrip(interval.unit, q))
    return (lo is None or v >= lo) and (hi is None or v <= hi)


class TestDomainsAreWellFormed:
    """Every chart reports a domain for exactly its own components."""

    @pytest.mark.parametrize("chart", CHARTS, ids=CHART_IDS)
    def test_keys_match_components(self, chart: cxc.AbstractChart) -> None:
        assert set(component_domains(chart)) == set(chart.components)

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


class TestRadial1DIsNotRadial:
    """`Radial1D` names its coordinate `r` but is a relabelled `Cart1D`.

    Its own docstring says "semantically equivalent to Cart1D but uses `r`
    instead of `x`", and `pt_map` agrees: the sign carries through. Reading a
    domain off the *name* would halve it -- the mirror image of the
    `Spherical3D` / `MathSpherical3D` case, where the name is the same but the
    meaning differs.
    """

    def test_domain_matches_cart1d(self) -> None:
        assert (
            component_domains(cxc.radial1d)["r"] == component_domains(cxc.cart1d)["x"]
        )

    def test_negative_r_round_trips(self) -> None:
        """The behaviour that makes a positive-only domain wrong."""
        there = cxc.pt_map({"x": u.Q(-5.0, "m")}, cxc.cart1d, cxc.radial1d)
        back = cxc.pt_map(there, cxc.radial1d, cxc.cart1d)
        assert float(u.ustrip("m", there["r"])) == pytest.approx(-5.0)
        assert float(u.ustrip("m", back["x"])) == pytest.approx(-5.0)

    def test_draws_reach_negative_values(self) -> None:
        seen_negative = False

        @given(point=cxst.cdicts(cxc.radial1d))
        @settings(
            max_examples=100, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            nonlocal seen_negative
            if float(u.ustrip("m", point["r"])) < 0:
                seen_negative = True

        collect()
        assert seen_negative


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


#: Physical truths about each coordinate, written out rather than derived from
#: `component_domains`, so that a wrong constant in that table cannot make the
#: test agree with it. Covers both spherical conventions, which is the case the
#: whole dispatch design exists for.
INVARIANTS = [
    pytest.param(cxc.sph3d, "r", "m", lambda v: v > 0, id="sph3d-r-positive"),
    pytest.param(
        cxc.sph3d, "theta", "rad", lambda v: 0 < v < math.pi, id="sph3d-theta-polar"
    ),
    pytest.param(
        cxc.math_sph3d,
        "phi",
        "rad",
        lambda v: 0 < v < math.pi,
        id="math_sph3d-phi-polar",
    ),
    pytest.param(
        cxc.math_sph3d,
        "theta",
        "rad",
        lambda v: -math.pi <= v <= math.pi,
        id="math_sph3d-theta-azimuth",
    ),
    pytest.param(cxc.polar2d, "r", "m", lambda v: v > 0, id="polar2d-r-positive"),
    pytest.param(cxc.cyl3d, "rho", "m", lambda v: v > 0, id="cyl3d-rho-positive"),
    pytest.param(
        cxc.lonlat_sph3d,
        "lat",
        "rad",
        lambda v: -math.pi / 2 < v < math.pi / 2,
        id="lonlat-lat-in-range",
    ),
    pytest.param(
        cxc.sph2, "theta", "rad", lambda v: 0 < v < math.pi, id="sph2-theta-polar"
    ),
]


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

    @pytest.mark.parametrize(
        ("chart", "component", "unit_str", "predicate"), INVARIANTS
    )
    def test_physical_invariant_holds(
        self,
        chart: cxc.AbstractChart,
        component: str,
        unit_str: str,
        predicate: Callable[[float], bool],
    ) -> None:
        """Draws satisfy the physics, stated *without* consulting the domains.

        `test_draws_are_in_domain` above checks draws against
        `component_domains` -- the same table `cdicts` generated from, so it
        stays green even if a constant in that table is wrong. It proves the
        strategy honours the domain, not that the domain is right. These
        assertions are the independent half: blank out `RADIAL` and they fail.
        """

        @given(point=cxst.cdicts(chart))
        @settings(
            max_examples=100, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            assert predicate(float(u.ustrip(unit_str, point[component])))

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


class TestMagnitudeFloorIsRadialOnly:
    """The floor is a length scale, so it must not reach a bounded angle."""

    def test_floor_does_not_move_a_polar_angle(self) -> None:
        """`magnitude=(0.5, 8)` must not shove theta 0.5 *radians* off the pole.

        Both `RADIAL` and `POLAR` start at zero, so a predicate keyed on that
        alone catches the colatitude too and couples an angle to what the
        caller meant as a length scale. Only `RADIAL` runs to infinity, and
        that is the discriminator.
        """
        thetas = []

        @given(point=cxst.cdicts(cxc.sph3d, magnitude=(0.5, 8.0)))
        @settings(
            max_examples=150, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            thetas.append(float(u.ustrip("rad", point["theta"])))

        collect()
        # Free to sit anywhere above its own margin, well below the 0.5 floor.
        assert min(thetas) < 0.5

    def test_floor_still_moves_the_radius(self) -> None:
        """The counterpart: the coordinate the floor is actually for."""

        @given(point=cxst.cdicts(cxc.sph3d, magnitude=(0.5, 8.0)))
        @settings(
            max_examples=150, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            assert float(u.ustrip("m", point["r"])) >= 0.5 * (1 - 1e-6)

        check()


class TestMappingElementsAreSafe:
    """Kwargs for the element strategy inherit the same safety defaults."""

    def test_no_nan_or_infinity_when_bounds_are_absent(self) -> None:
        """The hole a caller-supplied mapping used to open.

        The domain bounds hide this whenever they are finite, so it only shows
        with an unconstrained component *and* `magnitude=None` -- which was 193
        non-finite draws in 300 before the defaults were applied.
        """
        values = []

        @given(point=cxst.cdicts(cxc.cart3d, elements={}, magnitude=None))
        @settings(
            max_examples=200, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def collect(point: dict) -> None:
            values.extend(float(u.ustrip("m", q)) for q in point.values())

        collect()
        assert all(math.isfinite(v) for v in values)

    def test_caller_keys_still_win(self) -> None:
        """The defaults fill gaps; they do not override an explicit choice."""

        @given(
            point=cxst.cdicts(cxc.cart3d, elements={"min_value": 1.0, "max_value": 2.0})
        )
        @settings(
            max_examples=100, deadline=None, suppress_health_check=list(HealthCheck)
        )
        def check(point: dict) -> None:
            for q in point.values():
                assert 1.0 <= float(u.ustrip("m", q)) <= 2.0

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
