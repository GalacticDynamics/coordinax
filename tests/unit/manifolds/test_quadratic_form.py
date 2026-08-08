"""Tests for the shared metric quadratic form.

The point of extracting this is that every magnitude-like verb contracts the
same `vᵀ G(p) v`, and a second copy of that contraction reliably ends up with a
subset of the unit checks. So the tests here pin two things: the relationship to
`norm` (it is exactly the square), and that the fiddly unit handling lives here
rather than in the caller.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp
import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.manifolds.quadratic_form import quadratic_form

ATOL = 1e-5


class TestRelationToNorm:
    """`norm` is the square root of the form, for a positive-definite metric."""

    @pytest.mark.parametrize(
        ("v", "chart", "want"),
        [
            ({"x": 3.0, "y": 4.0, "z": 0.0}, cxc.cart3d, 25.0),
            ({"x": 1.0, "y": 0.0, "z": 0.0}, cxc.cart3d, 1.0),
            ({"x": 0.0, "y": 0.0, "z": 0.0}, cxc.cart3d, 0.0),
            ({"x": -5.0, "y": 12.0, "z": 0.0}, cxc.cart3d, 169.0),
        ],
    )
    def test_form_is_norm_squared(self, v, chart, want):
        vq = {k: u.Q(val, "m") for k, val in v.items()}
        at = {k: u.Q(0.0, "m") for k in v}

        got = quadratic_form(vq, chart, at=at)
        assert float(got.ustrip("m2")) == pytest.approx(want, abs=ATOL)

        n = cxm.norm(vq, chart, at=at)
        assert float(n.ustrip("m")) ** 2 == pytest.approx(want, abs=ATOL)

    def test_agrees_on_a_curved_metric(self):
        """Also holds where the metric actually varies with the base point."""
        at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
        v = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(1.0, "rad/s")}
        metric = cxm.RoundMetric(ndim=2)

        form = quadratic_form(v, cxc.sph2, at=at)
        n = cxm.norm(v, metric, cxc.sph2, at=at)
        assert float(u.ustrip("rad2/s2", form)) == pytest.approx(
            float(n.ustrip("rad/s")) ** 2, abs=ATOL
        )


class TestDefinedWhereNormIsNot:
    """The reason it exists: no square root, so an indefinite metric is fine."""

    AT4: ClassVar = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}

    @staticmethod
    def _v4(ct, x):
        return {
            "ct": u.Q(ct, "m"),
            "x": u.Q(x, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }

    @pytest.mark.parametrize(
        ("ct", "x", "want"), [(5.0, 1.0, -24.0), (1.0, 5.0, 24.0), (3.0, 3.0, 0.0)]
    )
    def test_indefinite_metric_gives_a_signed_value(self, ct, x, want):
        got = quadratic_form(self._v4(ct, x), cxc.minkowskict, at=self.AT4)
        assert float(got.ustrip("m2")) == pytest.approx(want, abs=ATOL)
        assert not bool(jnp.isnan(got.ustrip("m2")))

    def test_norm_still_refuses_the_same_input(self):
        """The form is permissive; the positive-definite guard stays on `norm`."""
        with pytest.raises(NotImplementedError, match=r"pseudo.*indefinite"):
            cxm.norm(
                self._v4(5.0, 1.0), cxm.MinkowskiMetric(), cxc.minkowskict, at=self.AT4
            )


class TestUnitHandlingLivesHere:
    """The checks a second copy of the contraction would have dropped."""

    def test_mixed_quantity_and_array_is_rejected(self):
        at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
        mixed = {"x": u.Q(3.0, "m"), "y": jnp.asarray(4.0), "z": u.Q(0.0, "m")}
        with pytest.raises(TypeError, match="mixed CDict"):
            quadratic_form(mixed, cxc.cart3d, at=at)

    def test_bare_arrays_require_usys(self):
        at = {"theta": jnp.asarray(jnp.pi / 2), "phi": jnp.asarray(0.0)}
        v = {"theta": jnp.asarray(1.0), "phi": jnp.asarray(0.0)}
        with pytest.raises(TypeError, match="usys"):
            quadratic_form(v, cxc.sph2, at=at)

    def test_bare_arrays_work_when_usys_is_given(self):
        at = {"theta": jnp.asarray(jnp.pi / 2), "phi": jnp.asarray(0.0)}
        v = {"theta": jnp.asarray(1.0), "phi": jnp.asarray(0.0)}
        got = quadratic_form(v, cxc.sph2, at=at, usys=u.unitsystems.si)
        assert float(got) == pytest.approx(1.0, abs=ATOL)

    @pytest.mark.parametrize("fname", ["norm", "interval", "quadratic_form"])
    def test_errors_name_the_calling_function(self, fname):
        """A caller of `norm` should not be told about `quadratic_form`."""
        at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
        mixed = {"x": u.Q(3.0, "m"), "y": jnp.asarray(4.0), "z": u.Q(0.0, "m")}
        with pytest.raises(TypeError, match=rf"^{fname}\(\)"):
            quadratic_form(mixed, cxc.cart3d, at=at, fname=fname)

    def test_mixed_unit_components_contract_correctly(self):
        """Per-component units must survive the contraction, not be flattened."""
        at = {
            "r": u.Q(5.0, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(0.0, "rad"),
        }
        v = {"r": u.Q(1.0, "m/s"), "theta": u.Q(0.0, "rad/s"), "phi": u.Q(0.0, "rad/s")}
        got = quadratic_form(v, cxc.sph3d, at=at)
        assert float(u.ustrip("m2/s2", got)) == pytest.approx(1.0, abs=ATOL)


class TestJAX:
    """Still traceable, since `norm` routes through it."""

    def test_jit(self):
        at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}

        @jax.jit
        def f(v):
            return quadratic_form(v, cxc.cart3d, at=at)

        v = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        assert float(f(v).ustrip("m2")) == pytest.approx(25.0, abs=ATOL)

    def test_vmap_batches(self):
        at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
        batch = {
            "x": u.Q(jnp.asarray([3.0, 1.0]), "m"),
            "y": u.Q(jnp.asarray([4.0, 0.0]), "m"),
            "z": u.Q(jnp.asarray([0.0, 0.0]), "m"),
        }
        got = jax.vmap(lambda v: quadratic_form(v, cxc.cart3d, at=at))(batch)
        assert qnp.allclose(
            got, u.Q(jnp.asarray([25.0, 1.0]), "m2"), atol=u.Q(ATOL, "m2")
        )
