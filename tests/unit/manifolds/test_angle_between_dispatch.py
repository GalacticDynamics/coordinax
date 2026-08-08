"""Tests for the angle_between() manifold API and wrappers."""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax.angles import AbstractAngle


def _m4(ct, x, y, z):
    """A Minkowski tangent-vector CDict."""
    return {
        "ct": jnp.array(ct),
        "x": jnp.array(x),
        "y": jnp.array(y),
        "z": jnp.array(z),
    }


class TestAngleBetweenEuclidean:
    """Tests for angle_between on Euclidean metrics and manifolds."""

    def test_cartesian_right_angle_returns_angle(self):
        metric = cxm.FlatMetric(2)
        at = {"x": u.Q(jnp.array(0), "m"), "y": u.Q(jnp.array(0), "m")}
        uvec = {"x": u.Q(jnp.array(1), "m"), "y": u.Q(jnp.array(0), "m")}
        vvec = {"x": u.Q(jnp.array(0), "m"), "y": u.Q(jnp.array(2), "m")}

        got = cxm.angle_between(metric, cxc.cart2d, uvec, vvec, at=at)

        assert isinstance(got, AbstractAngle)
        assert jnp.allclose(u.ustrip("rad", got), jnp.pi / 2, atol=1e-6)


class TestAngleBetweenFailureModes:
    """Tests for invalid inputs and unsupported metrics."""

    def test_zero_norm_vector_raises_value_error(self):
        metric = cxm.FlatMetric(2)
        at = {"x": jnp.array(0), "y": jnp.array(0)}
        zero = {"x": jnp.array(0), "y": jnp.array(0)}
        other = {"x": jnp.array(1), "y": jnp.array(0)}

        with pytest.raises(ValueError, match="zero"):
            cxm.angle_between(metric, cxc.cart2d, zero, other, at=at)

    def test_spacelike_pair_in_minkowski_has_an_ordinary_angle(self):
        """The metric on a spacelike 2-plane is positive-definite.

        This used to be rejected wholesale, on the metric's signature. The
        condition is really on the plane the two vectors span, not the metric.
        """
        at = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}
        xhat = _m4(0.0, 1.0, 0.0, 0.0)
        yhat = _m4(0.0, 0.0, 1.0, 0.0)
        got = cxm.angle_between(cxc.minkowskict, xhat, yhat, at=at)
        assert float(u.ustrip("rad", got)) == pytest.approx(jnp.pi / 2, abs=1e-5)

    def test_spacelike_pair_at_45_degrees(self):
        at = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}
        got = cxm.angle_between(
            cxc.minkowskict, _m4(0.0, 1.0, 0.0, 0.0), _m4(0.0, 1.0, 1.0, 0.0), at=at
        )
        assert float(u.ustrip("rad", got)) == pytest.approx(jnp.pi / 4, abs=1e-5)

    def test_two_timelike_vectors_are_a_hyperbolic_angle_not_a_circular_one(self):
        """`arccos` would clip to pi and report no relative motion.

        For two timelike vectors ``g(u,v)/sqrt(g(u,u) g(v,v))`` has magnitude
        >= 1 by the reverse Cauchy-Schwarz inequality, so it is a ``cosh``, not
        a ``cos``. The invariant is the relative rapidity.
        """
        at = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}
        obs = _m4(1.0, 0.0, 0.0, 0.0)
        moving = _m4(1.25, 0.75, 0.0, 0.0)  # beta = 0.6, gamma = 1.25
        with pytest.raises(ValueError, match="timelike"):
            cxm.angle_between(cxc.minkowskict, obs, moving, at=at)

    def test_mixed_causal_types_are_rejected(self):
        """``g(u,u) g(v,v) < 0`` makes the denominator imaginary."""
        at = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}
        with pytest.raises(ValueError, match="timelike and a spacelike"):
            cxm.angle_between(
                cxc.minkowskict, _m4(1.0, 0.0, 0.0, 0.0), _m4(0.0, 1.0, 0.0, 0.0), at=at
            )

    def test_null_vector_is_rejected(self):
        at = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}
        with pytest.raises(ValueError, match="null"):
            cxm.angle_between(
                cxc.minkowskict, _m4(1.0, 1.0, 0.0, 0.0), _m4(0.0, 1.0, 0.0, 0.0), at=at
            )

    def test_spacelike_pair_spanning_a_lorentzian_plane_is_rejected(self):
        """Both spacelike is *not* sufficient — the span can still be timelike.

        ``u = (0,1,0,0)`` and ``v = (1,2,0,0)`` are each spacelike, but their
        span contains the timelike ``(1,0,0,0)``, and ``|cos| = 1.15 > 1``.
        Clipping would silently report a 0 angle.
        """
        at = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}
        with pytest.raises(ValueError, match="span a plane that is not"):
            cxm.angle_between(
                cxc.minkowskict, _m4(0.0, 1.0, 0.0, 0.0), _m4(1.0, 2.0, 0.0, 0.0), at=at
            )


class TestAngleBetweenJAX:
    """Tests for JAX compatibility of angle_between."""

    def test_jit(self):
        metric = cxm.RoundMetric(ndim=2)

        @jax.jit
        def compute(theta):
            at = {"theta": theta, "phi": jnp.array(0)}
            uvec = {"theta": jnp.array(1), "phi": jnp.array(0)}
            vvec = {"theta": jnp.array(1), "phi": jnp.array(1)}
            return u.ustrip(
                "rad", cxm.angle_between(metric, cxc.sph2, uvec, vvec, at=at)
            )

        got = compute(jnp.array(jnp.pi / 2))
        assert jnp.allclose(got, jnp.pi / 4, atol=1e-6)

    def test_vmap_values(self):
        metric = cxm.RoundMetric(ndim=2)
        thetas = jnp.array([jnp.pi / 6, jnp.pi / 4, jnp.pi / 2])

        def compute(theta):
            at = {"theta": theta, "phi": jnp.array(0)}
            uvec = {"theta": jnp.array(1), "phi": jnp.array(0)}
            vvec = {"theta": jnp.array(1), "phi": jnp.array(1)}
            return u.ustrip(
                "rad", cxm.angle_between(metric, cxc.sph2, uvec, vvec, at=at)
            )

        got = jax.vmap(compute)(thetas)
        expected = jnp.arccos(1 / jnp.sqrt(1 + jnp.sin(thetas) ** 2))
        assert jnp.allclose(got, expected, atol=1e-6)


class TestTransformedCodepathsDoNotSilentlyClip:
    """The eager guard cannot fire under tracing, so the mask must.

    Regression: with only the eager check, `jax.jit` reached the `clip` and
    returned pi rad for two timelike vectors -- reporting "anti-parallel" for
    two observers in relative motion -- and 0 rad for a spacelike pair spanning
    a Lorentzian plane. Both are the wrong-but-real failure this whole change
    exists to remove, so they must be covered under transformation too.
    """

    AT: ClassVar = {k: jnp.array(0.0) for k in ("ct", "x", "y", "z")}

    @staticmethod
    def _angle(uvec, vvec, at):
        return cxm.angle_between(cxc.minkowskict, uvec, vvec, at=at)

    @pytest.mark.parametrize(
        ("label", "uvec", "vvec"),
        [
            ("timelike-timelike", (1.0, 0.0, 0.0, 0.0), (1.25, 0.75, 0.0, 0.0)),
            (
                "spacelike pair, Lorentzian span",
                (0.0, 1.0, 0.0, 0.0),
                (1.0, 2.0, 0.0, 0.0),
            ),
            ("mixed causal types", (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
            ("null", (1.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ],
    )
    def test_jit_yields_nan_not_a_plausible_angle(self, label, uvec, vvec):
        del label
        fn = jax.jit(lambda a, b: self._angle(a, b, self.AT))
        got = fn(_m4(*uvec), _m4(*vvec))
        assert bool(jnp.isnan(u.ustrip("rad", got))), f"expected nan, got {got}"

    def test_jit_still_returns_the_angle_when_it_is_defined(self):
        fn = jax.jit(lambda a, b: self._angle(a, b, self.AT))
        got = fn(_m4(0.0, 1.0, 0.0, 0.0), _m4(0.0, 0.0, 1.0, 0.0))
        assert float(u.ustrip("rad", got)) == pytest.approx(jnp.pi / 2, abs=1e-5)

    def test_vmap_marks_only_the_invalid_entries(self):
        """A mixed batch must not poison the valid entries, nor hide the bad."""
        us = {
            k: jnp.stack([_m4(1.0, 0.0, 0.0, 0.0)[k], _m4(0.0, 1.0, 0.0, 0.0)[k]])
            for k in ("ct", "x", "y", "z")
        }
        vs = {
            k: jnp.stack([_m4(1.25, 0.75, 0.0, 0.0)[k], _m4(0.0, 0.0, 1.0, 0.0)[k]])
            for k in ("ct", "x", "y", "z")
        }
        got = jax.vmap(lambda a, b: self._angle(a, b, self.AT))(us, vs)
        vals = u.ustrip("rad", got)
        assert bool(jnp.isnan(vals[0]))
        assert float(vals[1]) == pytest.approx(jnp.pi / 2, abs=1e-5)

    def test_riemannian_jit_is_untouched(self):
        """The mask must not introduce nan on the positive-definite path."""
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        xh = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        yh = {"x": jnp.array(0.0), "y": jnp.array(1.0), "z": jnp.array(0.0)}
        fn = jax.jit(lambda a, b: cxm.angle_between(cxc.cart3d, a, b, at=at))
        assert float(u.ustrip("rad", fn(xh, yh))) == pytest.approx(jnp.pi / 2, abs=1e-5)
