"""The `Point` overload of `chord_distance`.

`geodesic_distance` has one; this is its counterpart. It cannot be written the
same way: that overload brings both operands into a Cartesian chart, and for a
chord that is exactly wrong -- a Euclidean manifold is its own ambient, so
`chord_distance` refuses it, and an intrinsic sphere chart has no global
Cartesian representation to convert to at all.
"""

import math

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.manifolds as cxm


def _sph2(theta: float, phi: float) -> cx.Point:
    return cx.Point(
        {"theta": u.Angle(theta, "rad"), "phi": u.Angle(phi, "rad")}, chart=cxc.sph2
    )


class TestChordDistancePoint:
    """Measured through the ambient space, not along the manifold."""

    def test_matches_the_manifold_level_call(self) -> None:
        p, q = _sph2(math.pi / 2, 0.0), _sph2(math.pi / 2, math.pi / 2)
        direct = cxm.chord_distance(cxc.sph2, p.data, q.data)
        assert float(cx.chord_distance(p, q)) == pytest.approx(float(direct))

    @pytest.mark.parametrize("dphi", [1e-3, 0.5, 1.0, 2.0, math.pi])
    def test_matches_the_analytic_chord(self, dphi: float) -> None:
        """On the unit sphere the chord is ``2 sin(dphi / 2)``."""
        p, q = _sph2(math.pi / 2, 0.0), _sph2(math.pi / 2, dphi)
        assert float(cx.chord_distance(p, q)) == pytest.approx(
            2 * math.sin(dphi / 2), abs=1e-9
        )

    def test_is_symmetric(self) -> None:
        p, q = _sph2(1.0, 0.4), _sph2(1.6, 1.2)
        assert float(cx.chord_distance(p, q)) == pytest.approx(
            float(cx.chord_distance(q, p))
        )

    def test_is_chart_invariant(self) -> None:
        """The operands need not share a chart, and the answer must not care."""
        p = _sph2(math.pi / 2, 0.0)
        q_sph = _sph2(math.pi / 2, math.pi / 2)
        q_ll = cx.Point(
            {"lon": u.Angle(jnp.pi / 2, "rad"), "lat": u.Angle(0.0, "rad")},
            chart=cxc.lonlat_sph2,
        )
        assert float(cx.chord_distance(p, q_ll)) == pytest.approx(
            float(cx.chord_distance(p, q_sph))
        )

    def test_differs_from_the_geodesic(self) -> None:
        """The two verbs must not be confusable: sqrt(2) against pi/2."""
        p, q = _sph2(math.pi / 2, 0.0), _sph2(math.pi / 2, math.pi / 2)
        assert float(cx.chord_distance(p, q)) == pytest.approx(math.sqrt(2), abs=1e-9)

    def test_flat_space_is_refused(self) -> None:
        """Euclidean is its own ambient; the error points at `geodesic_distance`."""
        a = cx.Point.from_([3.0, 0.0, 0.0], "m")
        b = cx.Point.from_([0.0, 4.0, 0.0], "m")
        with pytest.raises(NotImplementedError, match="own ambient"):
            cx.chord_distance(a, b)

    def test_cross_frame_is_refused(self) -> None:
        """Frame-strict, as `geodesic_distance` is."""
        p = _sph2(math.pi / 2, 0.0)
        q = _sph2(math.pi / 2, math.pi / 2)
        moved = cx.Point(q.data, chart=q.chart, frame=cx.frames.Alice())
        with pytest.raises(ValueError, match="different frames"):
            cx.chord_distance(p, moved)

    def test_is_evaluated_elementwise_over_batch(self) -> None:
        """Chord distance is evaluated element-wise over the batch."""
        p = cx.Point(
            {
                "theta": u.Angle(jnp.full(2, math.pi / 2), "rad"),
                "phi": u.Angle(jnp.zeros(2), "rad"),
            },
            chart=cxc.sph2,
        )
        q = cx.Point(
            {
                "theta": u.Angle(jnp.full(2, math.pi / 2), "rad"),
                "phi": u.Angle(jnp.array([math.pi / 2, math.pi]), "rad"),
            },
            chart=cxc.sph2,
        )
        d = cx.chord_distance(p, q)
        assert float(d[0]) == pytest.approx(math.sqrt(2), abs=1e-9)
        assert float(d[1]) == pytest.approx(2.0, abs=1e-9)

    def test_different_manifolds_is_refused(self) -> None:
        """A 2-sphere point and a Euclidean point share no manifold to measure on."""
        p = _sph2(math.pi / 2, 0.0)
        q = cx.Point.from_([0.0, 4.0, 0.0], "m")
        with pytest.raises(ValueError, match="different manifolds"):
            cx.chord_distance(p, q)
