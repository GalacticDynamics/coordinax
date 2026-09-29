r"""Tests for `coordinax.charts.is_affine_transition`.

A transition is affine exactly when $\partial^2\psi \equiv 0$, which is the
condition under which an order-2 fibre may be carried between charts without
its velocity. The predicate is therefore a correctness claim, not a hint: a
wrong `True` silently drops the curvature term, while a wrong `False` only
costs a prolongation nobody needed.
"""

__all__: tuple[str, ...] = ()


import pytest

import unxt as u

import coordinax.charts as cxc
from coordinax.transforms._src.actions.prolong import prolong_point_map

T = u.unit("Myr")

# Two well-separated interior points are not enough on their own: a velocity
# parallel to the position is purely radial, and D2psi(v, v) vanishes along it
# even for cart3d -> sph3d. These directions are deliberately not parallel to
# any sample below.
DIRECTIONS = ((0.31, -0.74, 0.52), (-0.83, 0.19, -0.46))

SAMPLES: dict[str, dict[str, u.Q]] = {
    "cart2d": {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc")},
    "polar2d": {"r": u.Q(2.0, "kpc"), "theta": u.Q(0.7, "rad")},
    "sph2": {"theta": u.Q(0.9, "rad"), "phi": u.Q(0.4, "rad")},
    "math_sph2": {"theta": u.Q(0.4, "rad"), "phi": u.Q(0.9, "rad")},
    "lonlat_sph2": {"lon": u.Q(25.0, "deg"), "lat": u.Q(40.0, "deg")},
    "loncoslat_sph2": {"lon_coslat": u.Q(20.0, "deg"), "lat": u.Q(40.0, "deg")},
    "cart3d": {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    "sph3d": {"r": u.Q(2.0, "kpc"), "theta": u.Q(0.9, "rad"), "phi": u.Q(0.4, "rad")},
    "math_sph3d": {
        "r": u.Q(2.0, "kpc"),
        "theta": u.Q(0.4, "rad"),
        "phi": u.Q(0.9, "rad"),
    },
    "lonlat_sph3d": {
        "lon": u.Q(25.0, "deg"),
        "lat": u.Q(40.0, "deg"),
        "distance": u.Q(2.0, "kpc"),
    },
    "loncoslat_sph3d": {
        "lon_coslat": u.Q(20.0, "deg"),
        "lat": u.Q(40.0, "deg"),
        "distance": u.Q(2.0, "kpc"),
    },
    "cyl3d": {"rho": u.Q(2.0, "kpc"), "phi": u.Q(0.4, "rad"), "z": u.Q(1.0, "kpc")},
    "cart1d": {"x": u.Q(0.7, "kpc")},
    "radial1d": {"r": u.Q(0.7, "kpc")},
}


def curvature(a: str, b: str) -> float:
    r"""Max $|\partial^2\psi(v, v)|$ over the probe directions.

    Measured, not derived: push a jet whose acceleration is zero through the
    transition and read the acceleration that comes out. Whatever is there is
    the curvature term and nothing else.
    """
    ca, cb = getattr(cxc, a), getattr(cxc, b)
    q = SAMPLES[a]
    zero = {k: u.Q(0.0, u.unit_of(x) / T**2) for k, x in q.items()}
    worst = 0.0
    for draw in DIRECTIONS:
        v = {
            k: u.Q(float(d), u.unit_of(x) / T)
            for d, (k, x) in zip(draw, q.items(), strict=False)
        }
        out = prolong_point_map(
            lambda d, _f=ca, _t=cb: cxc.pt_map(d, _f, _t), {0: q, 1: v, 2: zero}
        )
        worst = max(
            worst, *(abs(float(u.ustrip(u.unit_of(x), x))) for x in out[2].values())
        )
    return worst


class TestTheClaimMatchesTheCurvature:
    """The predicate must agree with the thing it claims about."""

    def test_it_never_claims_affine_when_it_is_not(self) -> None:
        """The unsafe direction: a false `True` loses the term in silence.

        Swept over every same-dimension pair rather than a list of known
        pairs, so a new chart added to `SAMPLES` is checked without anyone
        remembering to check it.
        """
        claimed = [
            (a, b)
            for a in SAMPLES
            for b in SAMPLES
            if a != b
            and len(getattr(cxc, a).components) == len(getattr(cxc, b).components)
            and cxc.is_affine_transition(getattr(cxc, a), getattr(cxc, b))
        ]
        assert claimed, "swept nothing -- the predicate answers False everywhere"
        assert {p for p in claimed if curvature(*p) > 1e-12} == set()

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("sph3d", "lonlat_sph3d"),
            ("lonlat_sph3d", "math_sph3d"),
            ("math_sph3d", "sph3d"),
            ("sph2", "lonlat_sph2"),
            ("lonlat_sph2", "math_sph2"),
            ("math_sph2", "sph2"),
            ("cart1d", "radial1d"),
            ("radial1d", "cart1d"),
        ],
    )
    def test_the_declared_relabellings_really_are(self, a, b) -> None:
        assert curvature(a, b) < 1e-12
        assert cxc.is_affine_transition(getattr(cxc, a), getattr(cxc, b))

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("sph3d", "loncoslat_sph3d"),
            ("sph2", "loncoslat_sph2"),
            ("cart2d", "polar2d"),
            ("cart3d", "sph3d"),
            ("cart3d", "cyl3d"),
        ],
    )
    def test_the_near_misses_are_refused(self, a, b) -> None:
        """`lon_coslat` shares the spherical base class and still curves.

        `cart3d -> sph3d` is here so the probe has to bite: a test that only
        ever saw affine pairs would pass on a predicate returning `True`
        unconditionally.
        """
        assert curvature(a, b) > 1e-6
        assert not cxc.is_affine_transition(getattr(cxc, a), getattr(cxc, b))

    def test_two_distinct_cartesian_charts_are_affine(self) -> None:
        """The flat-to-flat branch, which needs two *different* flat charts.

        `cart3d` is the only Cartesian 3-D chart, so the same-chart branch
        answers every 3-D case first; the 4-D spacetime pair is what actually
        exercises this one.
        """
        assert cxc.galileanct != cxc.minkowskict
        assert cxc.is_affine_transition(cxc.galileanct, cxc.minkowskict)

    def test_a_chart_with_itself_is_the_identity(self) -> None:
        for name in ("cart3d", "sph3d", "loncoslat_sph3d", "cyl3d"):
            assert cxc.is_affine_transition(getattr(cxc, name), getattr(cxc, name))


class TestItIsAnExtensionPoint:
    """A chart family must be able to declare its own relabellings.

    The point of dispatching this rather than keeping a list: `coordinaxs.astro`
    or any downstream package defines charts `coordinax` has never heard of,
    and without a way to say "these two of mine are relabellings" they are
    permanently read as curvilinear -- paying for a prolongation on every
    order-2 conversion, and refused outright when a bundle has no velocity.
    """

    def test_it_is_dispatched_on_the_public_api(self) -> None:
        assert hasattr(cxc.is_affine_transition, "methods")
        assert hasattr(cxc.is_affine_transition, "dispatch")

    def test_a_family_registers_from_its_own_package(self) -> None:
        """Proof the mechanism works across packages, not just in principle.

        The 2-sphere charts live in `_src/spherical` and declare their own
        affinity there, exactly as a downstream package would. If that rule
        resolved from `_src/charts` instead, the registration would be
        centralised after all and the extension point unproven.
        """
        f = cxc.is_affine_transition
        f._resolve_pending_registrations()

        impl_2s, _ = f.resolve_method((cxc.sph2, cxc.lonlat_sph2))
        impl_3d, _ = f.resolve_method((cxc.sph3d, cxc.lonlat_sph3d))
        assert impl_2s.__module__ != impl_3d.__module__
        assert impl_2s.__module__.endswith("spherical.register_charts")

    def test_an_undeclared_pair_falls_back_to_the_safe_answer(self) -> None:
        """Unknown means "assume it curves", which is slow, never wrong."""
        assert not cxc.is_affine_transition(cxc.cyl3d, cxc.sph3d)
