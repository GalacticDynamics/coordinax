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
import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from coordinax._src.manifolds.quadratic_form import (
    _contract,
    bilinear_form,
    quadratic_form,
)
from coordinax._src.metric.matrix import DiagonalMetric

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


class TestDiagonalFastPath:
    """`DiagonalMetric` contracts in O(n); it must agree with the O(n^2) form.

    `metric_matrix` returns a `DiagonalMetric` for every metric the library
    ships in its canonical chart, and that type exists so the full matrix need
    never be materialised. `to_dense()` discards the structure, so the fast path
    is the common one — and its whole job is to be indistinguishable from the
    dense answer.
    """

    CASES: ClassVar = [
        (
            "cart3d",
            cxc.cart3d,
            {"x": (0.0, "m"), "y": (0.0, "m"), "z": (0.0, "m")},
            {"x": (3.0, "m"), "y": (4.0, "m"), "z": (0.0, "m")},
        ),
        (
            "minkowskict",
            cxc.minkowskict,
            {"ct": (0.0, "m"), "x": (0.0, "m"), "y": (0.0, "m"), "z": (0.0, "m")},
            {"ct": (5.0, "m"), "x": (1.0, "m"), "y": (0.0, "m"), "z": (0.0, "m")},
        ),
        (
            "sph2 (curved)",
            cxc.sph2,
            {"theta": (jnp.pi / 2, "rad"), "phi": (0.0, "rad")},
            {"theta": (1.0, "rad/s"), "phi": (1.0, "rad/s")},
        ),
        (
            "sph3d (mixed units)",
            cxc.sph3d,
            {"r": (5.0, "m"), "theta": (jnp.pi / 2, "rad"), "phi": (0.0, "rad")},
            {"r": (1.0, "m/s"), "theta": (0.0, "rad/s"), "phi": (0.0, "rad/s")},
        ),
    ]

    @pytest.mark.parametrize(("label", "chart", "at_spec", "v_spec"), CASES)
    def test_agrees_with_the_dense_contraction(self, label, chart, at_spec, v_spec):
        del label
        at = {k: u.Q(val, un) for k, (val, un) in at_spec.items()}
        v = {k: u.Q(val, un) for k, (val, un) in v_spec.items()}
        mm = cxmapi.metric_matrix(chart.M, at, chart)
        assert isinstance(mm, DiagonalMetric), "case must exercise the fast path"

        v_qm = cxcapi.carray(v, chart.components)
        fast = _contract(mm, v_qm, v_qm)
        dense = _contract(mm.to_dense(), v_qm, v_qm)  # forces the generic fallback

        assert u.unit_of(fast) == u.unit_of(dense)
        assert qnp.allclose(fast, dense, atol=u.Q(1e-8, u.unit_of(dense)))

    @pytest.mark.parametrize(
        ("label", "chart", "at_spec", "v_spec"),
        [c for c in CASES if c[0] != "sph3d (mixed units)"],
    )
    def test_bare_array_branch_agrees_too(self, label, chart, at_spec, v_spec):
        """The stacked-array path has its own overload; it must match as well.

        ``sph3d`` is excluded: a unitless vector against a unitful (mixed-unit)
        diagonal is unsupported on *both* paths, before this change and after,
        so there is no agreement to assert.
        """
        del label
        at = {k: u.Q(val, un) for k, (val, un) in at_spec.items()}
        v_arr = {k: jnp.asarray(val) for k, (val, _) in v_spec.items()}
        mm = cxmapi.metric_matrix(chart.M, at, chart)
        stacked = jnp.stack([v_arr[k] for k in chart.components], axis=-1)

        fast = _contract(mm, stacked, stacked)
        dense = _contract(mm.to_dense(), stacked, stacked)
        assert jnp.allclose(jnp.asarray(fast), jnp.asarray(dense), atol=1e-8)

    def test_batched_arrays_reduce_over_components_not_the_batch(self):
        """``sum(..., axis=-1)`` rather than ``@``, which would eat the batch axis."""
        at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
        mm = cxmapi.metric_matrix(cxc.cart3d.M, at, cxc.cart3d)
        batch = jnp.asarray([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 12.0]])
        got = _contract(mm, batch, batch)
        assert got.shape == (3,)
        assert jnp.allclose(got, jnp.asarray([25.0, 1.0, 169.0]), atol=1e-8)

    def test_norm_is_unchanged_end_to_end(self):
        """The optimisation is invisible from the public verb."""
        at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
        v = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        assert float(cxm.norm(v, cxc.cart3d, at=at).ustrip("m")) == pytest.approx(
            5.0, abs=ATOL
        )


class TestBilinearForm:
    """`quadratic_form` is the ``u is v`` case; `angle_between` needs the rest."""

    AT: ClassVar = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    XHAT: ClassVar = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    YHAT: ClassVar = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")}
    V: ClassVar = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}

    def test_orthogonal_vectors_contract_to_zero(self):
        got = bilinear_form(self.XHAT, self.YHAT, cxc.cart3d, at=self.AT)
        assert float(got.ustrip("m2")) == pytest.approx(0.0, abs=ATOL)

    def test_reduces_to_quadratic_form_when_both_vectors_are_the_same(self):
        """The defining relationship between the two spellings."""
        bi = bilinear_form(self.V, self.V, cxc.cart3d, at=self.AT)
        qu = quadratic_form(self.V, cxc.cart3d, at=self.AT)
        assert float(bi.ustrip("m2")) == pytest.approx(float(qu.ustrip("m2")), abs=ATOL)

    def test_is_symmetric(self):
        """As the metric is."""
        a = bilinear_form(self.V, self.XHAT, cxc.cart3d, at=self.AT)
        b = bilinear_form(self.XHAT, self.V, cxc.cart3d, at=self.AT)
        assert float(a.ustrip("m2")) == pytest.approx(float(b.ustrip("m2")), abs=ATOL)

    def test_indefinite_metric_gives_a_signed_value(self):
        at4 = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
        t = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(0.0, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        got = bilinear_form(t, t, cxc.minkowskict, at=at4)
        assert float(got.ustrip("m2")) == pytest.approx(-1.0, abs=ATOL)

    def test_mixed_cdict_is_rejected_in_either_argument(self):
        mixed = {"x": u.Q(3.0, "m"), "y": jnp.asarray(4.0), "z": u.Q(0.0, "m")}
        with pytest.raises(TypeError, match="mixed CDict"):
            bilinear_form(mixed, self.XHAT, cxc.cart3d, at=self.AT)
        with pytest.raises(TypeError, match="mixed CDict"):
            bilinear_form(self.XHAT, mixed, cxc.cart3d, at=self.AT)

    def test_cross_argument_mixing_quantity_and_bare_array_is_rejected(self):
        """One Quantity CDict and one bare-array CDict across arguments is rejected.

        Without this check, the code would pass `_prepare()` but fail in `_contract()`
        with a plum dispatch error, since there's no overload for
        `(AbstractMetricMatrix, QuantityMatrix, Array)` or vice versa.
        """
        bare = {"x": jnp.asarray(1.0), "y": jnp.asarray(0.0), "z": jnp.asarray(0.0)}
        with pytest.raises(TypeError, match="consistently either Quantity"):
            bilinear_form(self.XHAT, bare, cxc.cart3d, at=self.AT, require_usys=False)
        with pytest.raises(TypeError, match="consistently either Quantity"):
            bilinear_form(bare, self.XHAT, cxc.cart3d, at=self.AT, require_usys=False)


class TestRequireUsysIsAPolicyNotAComputation:
    """Whether bare arrays need `usys` depends on the *verb*, not the primitive.

    `norm` and ``interval`` return a value whose unit is derived from the
    inputs, so demanding a unit system is a meaningful contract.
    `angle_between` returns a dimensionless ratio in which every unit cancels,
    so the same demand would be ceremony -- and would break callers that have
    always passed bare arrays.
    """

    BARE: ClassVar = {"theta": jnp.asarray(1.0), "phi": jnp.asarray(0.0)}
    AT: ClassVar = {"theta": jnp.asarray(jnp.pi / 2), "phi": jnp.asarray(0.0)}

    def test_default_demands_usys(self):
        with pytest.raises(TypeError, match="usys"):
            quadratic_form(self.BARE, cxc.sph2, at=self.AT)

    def test_opting_out_allows_bare_arrays(self):
        got = quadratic_form(self.BARE, cxc.sph2, at=self.AT, require_usys=False)
        assert float(got) == pytest.approx(1.0, abs=ATOL)

    def test_angle_between_accepts_bare_arrays(self):
        """Regression: migrating onto the shared primitive must not tighten this."""
        at = {"x": jnp.asarray(0.0), "y": jnp.asarray(0.0), "z": jnp.asarray(0.0)}
        xh = {"x": jnp.asarray(1.0), "y": jnp.asarray(0.0), "z": jnp.asarray(0.0)}
        yh = {"x": jnp.asarray(0.0), "y": jnp.asarray(1.0), "z": jnp.asarray(0.0)}
        got = cxm.angle_between(cxc.cart3d, xh, yh, at=at)
        assert float(u.ustrip("rad", got)) == pytest.approx(jnp.pi / 2, abs=1e-5)

    def test_angle_between_still_rejects_a_mixed_cdict(self):
        """It gains the check it never had, which is the point of sharing."""
        at = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        mixed = {"x": u.Q(1.0, "m"), "y": jnp.asarray(0.0), "z": u.Q(0.0, "m")}
        good = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")}
        with pytest.raises(TypeError, match="mixed CDict"):
            cxm.angle_between(cxc.cart3d, mixed, good, at=at)
