"""Unit tests for coordinax.distances.DistanceModulus using hypothesis strategies."""

import decimal

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import plum
import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.distances as cxd
import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro as cxastrost


class TestDistanceModulusConstruction:
    """Tests for DistanceModulus construction and basic properties."""

    @given(dm=cxastrost.distance_moduli())
    def test_unit_is_mag(self, dm: cxastro.DistanceModulus) -> None:
        """Distance moduli always have unit 'mag'."""
        assert dm.unit == u.unit("mag")

    def test_invalid_unit_raises(self) -> None:
        """DistanceModulus with non-mag unit raises ValueError."""
        with pytest.raises(ValueError, match="magnitude"):
            cxastro.DistanceModulus(15, "kpc")


class TestDistanceModulusArithmetic:
    """Tests for arithmetic operations on DistanceModulus."""

    @given(dm=cxastrost.distance_moduli())
    def test_add_distance_moduli(self, dm: cxastro.DistanceModulus) -> None:
        """DistanceModulus + DistanceModulus returns DistanceModulus."""
        result = dm + dm
        assert isinstance(result, cxastro.DistanceModulus)

    @given(dm=cxastrost.distance_moduli())
    def test_sub_distance_moduli(self, dm: cxastro.DistanceModulus) -> None:
        """DistanceModulus - DistanceModulus returns DistanceModulus with zero."""
        result = dm - dm
        assert isinstance(result, cxastro.DistanceModulus)
        assert jnp.allclose(result.value, 0)

    @given(dm=cxastrost.distance_moduli())
    def test_scalar_mul(self, dm: cxastro.DistanceModulus) -> None:
        """Scalar multiplication returns DistanceModulus."""
        result = 2 * dm
        assert isinstance(result, cxastro.DistanceModulus)
        assert jnp.allclose(result.value, 2 * dm.value)


class TestDistanceModulusConversionProperties:
    """Tests for DistanceModulus conversion properties."""

    @given(dm=cxastrost.distance_moduli(elements={"min_value": -5, "max_value": 25}))
    @settings(deadline=None)
    def test_distance_property(self, dm: cxastro.DistanceModulus) -> None:
        """.distance property returns a Distance."""
        assert isinstance(dm.distance, cxd.Distance)


class TestDistanceModulusPlumConvert:
    """Tests for plum.convert with DistanceModulus."""

    @given(dm=cxastrost.distance_moduli())
    def test_convert_to_quantity(self, dm: cxastro.DistanceModulus) -> None:
        """Can convert DistanceModulus to Quantity."""
        q = plum.convert(dm, u.Q)
        assert isinstance(q, u.Q)
        assert q.unit is dm.unit
        assert q.value is dm.value

    @given(dm=cxastrost.distance_moduli(elements={"min_value": -5, "max_value": 25}))
    @settings(deadline=None)
    def test_convert_to_distance(self, dm: cxastro.DistanceModulus) -> None:
        """Can convert DistanceModulus to Distance."""
        d = plum.convert(dm, cxd.Distance)
        assert isinstance(d, cxd.Distance)

    @given(dm=cxastrost.distance_moduli(elements={"min_value": -5, "max_value": 25}))
    @settings(deadline=None)
    def test_convert_to_parallax(self, dm: cxastro.DistanceModulus) -> None:
        """Can convert DistanceModulus to Parallax."""
        plx = plum.convert(dm, cxastro.Parallax)
        assert isinstance(plx, cxastro.Parallax)


class TestDistanceModulusAccuracyNearZeroPoint:
    """`dm = 5 log10(d/10pc)` is evaluated about its zero point.

    The algebraically equal `5*log10(d_pc) - 5` loses absolute precision for
    d ~ 10 pc, where dm -> 0: the multiply-by-5 lands while log10 ~ 1, and the
    exact `- 5` that follows cannot recover what it cost.

    These assert in **float32**. The effect is a few eps, so under the float64
    the suite normally runs (`JAX_ENABLE_X64=1`) both forms agree to ~1e-16 and
    no reachable tolerance separates them -- a float64 test here would pass
    whichever form is installed and guard nothing. JAX preserves input dtype,
    so passing float32 in is enough.

    Bounds below were measured through this exact code path (not an idealised
    snippet -- `ustrip` contributes) and each sits strictly between the two
    forms' error at that point, so reverting the implementation fails here.
    """

    #: (d [pc], bound, measured error of the subtractive form at that d).
    NEAR_ZERO_POINT: ClassVar = [
        pytest.param(9.9, 1.0e-7, 2.28e-7, id="9.9pc"),
        pytest.param(9.999, 1.0e-7, 1.92e-7, id="9.999pc"),
        pytest.param(10.001, 1.0e-7, 2.14e-7, id="10.001pc"),
        pytest.param(10.01, 1.2e-7, 3.51e-7, id="10.01pc"),
        pytest.param(10.1, 2.0e-7, 4.48e-7, id="10.1pc"),
    ]

    #: Worst absolute error over the sampled band, per form (measured).
    WORST_ERROR_BOUND = 2.0e-7  # this form: 1.37e-7; subtractive form: 4.48e-7

    @staticmethod
    def _exact_dm(d_pc: float) -> float:
        """Dm for the exact value the float32 holds, at 60 decimal digits."""
        with decimal.localcontext() as ctx:  # don't leak `prec` to other tests
            ctx.prec = 60
            d = decimal.Decimal(float(np.float32(d_pc)))
            return float(5 * (d / decimal.Decimal(10)).ln() / decimal.Decimal(10).ln())

    @classmethod
    def _abs_error(cls, d_pc: float) -> float:
        got = cxastro.DistanceModulus.from_(u.Q(np.float32(d_pc), "pc")).ustrip("mag")
        assert got.dtype == np.float32, "float32 lost; this test would prove nothing"
        return abs(float(got) - cls._exact_dm(d_pc))

    @pytest.mark.parametrize(("d_pc", "bound", "subtractive_error"), NEAR_ZERO_POINT)
    def test_float32_absolute_error(
        self, d_pc: float, bound: float, subtractive_error: float
    ) -> None:
        """Error stays under a bound the subtractive form cannot meet."""
        assert self._abs_error(d_pc) < bound
        # Guard the guard: a bound the old form also met would test nothing.
        assert bound < subtractive_error

    def test_worst_error_across_the_band(self) -> None:
        """Worst-case error over the band beats the subtractive form's 4.48e-7."""
        worst = max(
            self._abs_error(d)
            for d in (9.0, 9.9, 9.99, 9.999, 10.001, 10.01, 10.1, 11.0, 50.0)
        )
        assert worst < self.WORST_ERROR_BOUND

    def test_zero_point_is_exactly_zero(self) -> None:
        """dm(10 pc) is exactly 0, not merely close to it."""
        got = cxastro.DistanceModulus.from_(u.Q(10.0, "pc")).ustrip("mag")
        assert float(got) == 0.0


class TestParametricFromDispatch:
    """`from_` routes ParametricQuantity by type (optional unxts.parametric)."""

    @pytest.mark.parametrize(("value", "unit"), [(1, "pc"), (1, "mas"), (1, "mag")])
    def test_matches_quantity_path(self, value: float, unit: str) -> None:
        """A ParametricQuantity gives the same result as a plain Quantity."""
        pq = pytest.importorskip("unxts.parametric").PQ(value, unit)
        got = cxastro.DistanceModulus.from_(pq, dtype=float)
        expected = cxastro.DistanceModulus.from_(u.Q(value, unit), dtype=float)
        assert got.unit == expected.unit
        assert jnp.allclose(got.value, expected.value)
