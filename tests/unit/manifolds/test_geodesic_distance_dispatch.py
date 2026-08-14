"""Tests for the geodesic_distance() manifold API dispatches."""

import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.manifolds as cxm

_NORTH = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
_OTHER = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.0, "rad")}


class TestGeodesicDistanceDispatches:
    """The manifold-level `geodesic_distance` accepts several input forms."""

    def test_chart_and_cdicts(self):
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        d = cxm.geodesic_distance(cxc.cart3d, a, b)
        assert isinstance(d, cx.Distance)
        assert bool(qnp.isclose(d.ustrip("m"), 5.0))

    def test_metric_chart_and_cdicts(self):
        metric = cxm.FlatMetric(3)
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        assert bool(
            qnp.isclose(
                cxm.geodesic_distance(metric, cxc.cart3d, a, b).ustrip("m"), 5.0
            )
        )

    def test_chart_and_packed_quantities(self):
        a = u.Q([3.0, 0.0, 0.0], "m")
        b = u.Q([0.0, 4.0, 0.0], "m")
        assert bool(
            qnp.isclose(cxm.geodesic_distance(cxc.cart3d, a, b).ustrip("m"), 5.0)
        )

    def test_chart_and_bare_arrays(self):
        a = jnp.array([3.0, 0.0, 0.0])
        b = jnp.array([0.0, 4.0, 0.0])
        assert bool(qnp.isclose(cxm.geodesic_distance(cxc.cart3d, a, b), 5.0))

    def test_all_forms_agree_with_the_point_overload(self):
        p = cx.Point.from_([3.0, 0.0, 0.0], "m")
        q = cx.Point.from_([0.0, 4.0, 0.0], "m")
        ref = cxm.geodesic_distance(p, q).ustrip("m")
        packed = cxm.geodesic_distance(
            cxc.cart3d, u.Q([3.0, 0.0, 0.0], "m"), u.Q([0.0, 4.0, 0.0], "m")
        )
        assert bool(qnp.isclose(packed.ustrip("m"), ref))

    def test_packed_quantity_is_unit_invariant(self):
        a = u.Q([3.0, 0.0, 0.0], "m")
        b = u.Q([0.0, 0.004, 0.0], "km")
        assert bool(
            qnp.isclose(cxm.geodesic_distance(cxc.cart3d, a, b).ustrip("m"), 5.0)
        )

    def test_batched_packed_quantities(self):
        a = u.Q([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "m")
        b = u.Q([[0.0, 4.0, 0.0], [0.0, 1.0, 0.0]], "m")
        d = cxm.geodesic_distance(cxc.cart3d, a, b).ustrip("m")
        assert bool(qnp.isclose(d[0], 5.0))
        assert bool(qnp.isclose(d[1], qnp.sqrt(2.0)))


class TestIndefiniteMetricGeodesicDistance:
    """`geodesic_distance` inherits `norm`'s guard instead of returning ``nan``.

    Regression: a timelike pair used to yield ``Distance(nan, 'm')`` while a
    spacelike pair yielded a plausible number, so the failure was invisible
    unless you happened to probe a timelike interval.
    """

    @staticmethod
    def _event(ct, x):
        return {
            "ct": u.Q(ct, "m"),
            "x": u.Q(x, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }

    @pytest.mark.parametrize(
        ("kind", "ct", "x"),
        [("timelike", 5.0, 1.0), ("spacelike", 1.0, 5.0), ("null", 3.0, 3.0)],
    )
    def test_raises_rather_than_returning_nan(self, kind, ct, x):
        del kind
        origin = self._event(0.0, 0.0)
        with pytest.raises(NotImplementedError, match=r"pseudo.*indefinite"):
            cxm.geodesic_distance(cxc.minkowskict, origin, self._event(ct, x))

    def test_euclidean_geodesic_distance_is_unaffected(self):
        """Positive control: the common Riemannian path is untouched."""
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        assert bool(
            qnp.isclose(cxm.geodesic_distance(cxc.cart3d, a, b).ustrip("m"), 5.0)
        )


class TestEmbeddedSphere:
    """An embedded sphere's geodesic is its radius times the central angle.

    `EmbeddedChart.M` is an `EmbeddedManifold`, so reaching this through a
    chart lands on the same rule; without it the call fell through to the
    "no geodesic implemented" refusal.
    """

    RADIUS = u.Q(2.0, "m")

    def _manifold(self):
        return cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=self.RADIUS),
        )

    @pytest.mark.parametrize(
        ("theta", "expected"), [(jnp.pi, 2 * jnp.pi), (jnp.pi / 2, jnp.pi)]
    )
    def test_arc_length_scales_with_the_radius(self, theta, expected):
        north = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
        other = {"theta": u.Angle(theta, "rad"), "phi": u.Angle(0.0, "rad")}
        got = cxm.geodesic_distance(self._manifold(), cxc.sph2, north, other)
        assert bool(qnp.isclose(got.ustrip("m"), expected, atol=1e-12))

    def test_reached_through_an_embedded_chart(self):
        chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=self.RADIUS))
        north = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
        south = {"theta": u.Angle(jnp.pi, "rad"), "phi": u.Angle(0.0, "rad")}
        got = cxm.geodesic_distance(chart, north, south)
        assert bool(qnp.isclose(got.ustrip("m"), 2 * jnp.pi, atol=1e-12))

    def test_refuses_an_embedding_without_a_closed_form(self):
        """Only the two-sphere embedding has one; the rest must say so."""
        embed_map = cxm.CustomEmbeddingMap(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_fn=lambda p, **kw: p,
            project_fn=lambda p, **kw: p,
        )
        M = cxm.EmbeddedManifold(intrinsic=cxm.S2, ambient=cxm.R3, embed_map=embed_map)
        north = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
        other = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.0, "rad")}
        with pytest.raises(NotImplementedError, match="only the two-sphere"):
            cxm.geodesic_distance(M, cxc.sph2, north, other)

    def test_any_intrinsic_chart_agrees(self):
        M = self._manifold()
        a = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.4, "rad")}
        b = {"theta": u.Angle(1.6, "rad"), "phi": u.Angle(1.2, "rad")}
        sph = cxm.geodesic_distance(M, cxc.sph2, a, b).ustrip("m")
        lonlat = cxm.geodesic_distance(
            M,
            cxc.lonlat_sph2,
            cxc.pt_map(a, cxc.sph2, cxc.lonlat_sph2),
            cxc.pt_map(b, cxc.sph2, cxc.lonlat_sph2),
        ).ustrip("m")
        assert bool(qnp.isclose(sph, lonlat, atol=1e-12))


class TestRefusals:
    """Refusing is the design decision here, so it is pinned like any other.

    The alternative -- the norm of the coordinate difference -- is asymmetric
    on a curved manifold and so is not a distance at all. A silent wrong number
    is worse than an error, so each of these must keep raising.
    """

    def test_refuses_a_sphere_that_is_not_the_two_sphere(self):
        """The closed form is `S2`-specific; `S1` and up must not borrow it."""
        with pytest.raises(NotImplementedError, match=r"only.*two-sphere"):
            cxm.geodesic_distance(cxm.S1, cxc.sph2, _NORTH, _OTHER)

    def test_refuses_a_manifold_with_no_rule(self):
        """The `AbstractManifold` fallback: no closed form, so no answer."""
        with pytest.raises(NotImplementedError, match="no geodesic distance"):
            cxm.geodesic_distance(cxm.NoManifold(), cxc.sph2, _NORTH, _OTHER)


class TestChordDistance:
    """The chord is the ambient straight line, not an approximate geodesic.

    Both are exact; they answer different questions. On a sphere of radius R
    separated by a central angle t, the geodesic is ``R t`` and the chord is
    ``2 R sin(t / 2)``.
    """

    @pytest.mark.parametrize("theta", [1e-3, 0.1, 1.0, 2.0, jnp.pi])
    def test_matches_the_analytic_chord(self, theta):
        a = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        b = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(theta, "rad")}
        got = jnp.asarray(cxm.chord_distance(cxc.sph2, a, b))
        assert bool(qnp.isclose(got, 2 * jnp.sin(theta / 2), atol=1e-12))

    def test_is_symmetric_and_chart_invariant(self):
        a = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.4, "rad")}
        b = {"theta": u.Angle(1.6, "rad"), "phi": u.Angle(1.2, "rad")}
        ab = jnp.asarray(cxm.chord_distance(cxc.sph2, a, b))
        ba = jnp.asarray(cxm.chord_distance(cxc.sph2, b, a))
        lonlat = jnp.asarray(
            cxm.chord_distance(
                cxc.lonlat_sph2,
                cxc.pt_map(a, cxc.sph2, cxc.lonlat_sph2),
                cxc.pt_map(b, cxc.sph2, cxc.lonlat_sph2),
            )
        )
        assert bool(qnp.isclose(ab, ba, atol=1e-14))
        assert bool(qnp.isclose(ab, lonlat, atol=1e-14))

    def test_differs_from_the_geodesic(self):
        """The two must not be confusable: at a quarter turn they differ by 10%."""
        a = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        b = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(jnp.pi / 2, "rad")}
        chord = jnp.asarray(cxm.chord_distance(cxc.sph2, a, b))
        arc = jnp.asarray(cxm.geodesic_distance(cxc.sph2, a, b).ustrip("rad"))
        assert bool(qnp.isclose(chord, jnp.sqrt(2.0), atol=1e-12))
        assert bool(qnp.isclose(arc, jnp.pi / 2, atol=1e-12))

    def test_embedded_sphere_carries_its_radius(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
        )
        north = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
        south = {"theta": u.Angle(jnp.pi, "rad"), "phi": u.Angle(0.0, "rad")}
        got = cxm.chord_distance(M, cxc.sph2, north, south)
        assert bool(qnp.isclose(got.ustrip("m"), 4.0, atol=1e-12))

    def test_flat_space_is_refused(self):
        """Its own ambient: the chord is the straight line `geodesic_distance` gives."""
        a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        with pytest.raises(NotImplementedError, match="own ambient"):
            cxm.chord_distance(cxc.cart3d, a, b)

    def test_refuses_a_manifold_with_no_embedding(self):
        """No ambient space to cut through, so there is no chord to return."""
        with pytest.raises(NotImplementedError, match="carries no embedding"):
            cxm.chord_distance(cxm.NoManifold(), cxc.sph2, _NORTH, _OTHER)

    def test_refuses_a_sphere_that_is_not_the_two_sphere(self):
        """`TwoSphereIn3D` is the only embedding wired up here."""
        with pytest.raises(NotImplementedError, match=r"only.*two-sphere"):
            cxm.chord_distance(cxm.S1, cxc.sph2, _NORTH, _OTHER)
