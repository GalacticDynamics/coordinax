"""Integration tests for coordinax.angles.Angle with quax and quaxed.

Key behavioral contracts verified here:

* ``quaxed.numpy`` arithmetic/unary ops (add, sub, mul, abs, neg, …) preserve
  the ``Angle`` type and unit.
* ``quaxed.numpy`` trig ops (sin, cos, tan) return a *dimensionless*
  ``Quantity``, **not** an ``Angle`` — consuming the angular dimension.
* ``quax.quaxify`` is required to use raw ``jax.numpy`` functions directly on
  an ``Angle``; without it, JAX raises ``TypeError``.
* JAX transforms (``jit``, ``vmap``, ``grad``) compose correctly with
  ``quaxed.numpy`` operations on ``Angle``.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import pytest
import quax
from hypothesis import given, settings, strategies as st

import quaxed.numpy as qnp
import unxt as u
from unxt.quantity import AbstractAngle

import coordinax.angles as cxa
import coordinax.hypothesis.main as cxst

# ---------------------------------------------------------------------------
# Reusable hypothesis strategies
# ---------------------------------------------------------------------------

# Bounded to [-3, 3] rad — covers more than a full turn without overflowing
# trig functions
_angle_rad = cxst.angles(
    unit="rad", elements=st.floats(min_value=-3, max_value=3, width=32)
)

# Narrower range for grad tests where we compare with cos(x): avoids
# cancellation near ±π/2
_angle_rad_trig = cxst.angles(
    unit="rad", elements=st.floats(min_value=-1, max_value=1, width=32)
)


class TestQuaxedUnary:
    """quaxed.numpy unary ops preserve the Angle type and unit."""

    # --- correctness on known values ---

    @pytest.mark.parametrize(
        ("value", "fn", "expected"),
        [
            (1.5, qnp.abs, 1.5),  # positive → unchanged
            (-1.5, qnp.abs, 1.5),  # negative → flipped
            (1.5, qnp.negative, -1.5),
            (1.7, qnp.floor, 1),
            (1.2, qnp.ceil, 2),
            (1.6, qnp.round, 2),
        ],
    )
    def test_known_value(self, value: float, fn: object, expected: float) -> None:
        """Each unary op returns the expected numeric result on a concrete input."""
        result = fn(cxa.Angle(value, "rad"))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, expected)

    # --- property: result is always an Angle with the same unit ---

    @pytest.mark.parametrize("fn", [qnp.abs, qnp.negative])
    @given(angle=cxst.angles())
    def test_preserves_type_and_unit(self, fn: object, angle: cxa.Angle) -> None:
        """Abs and negative always return an Angle with the original unit."""
        result = fn(angle)
        assert isinstance(result, cxa.Angle)
        assert result.unit == angle.unit

    @given(angle=cxst.angles())
    def test_abs_nonneg(self, angle: cxa.Angle) -> None:
        """qnp.abs returns a non-negative value for every input."""
        assert jnp.all(qnp.abs(angle).value >= 0)


class TestQuaxedTrig:
    """Trig ops consume the angular dimension → dimensionless Quantity, not Angle."""

    # Both "returns the right type" and "returns the right value" are checked
    # in one parametrized test so we avoid duplicating the fixture setup.
    @pytest.mark.parametrize(
        ("fn", "angle_val", "expected"),
        [(qnp.sin, jnp.pi / 2, 1), (qnp.cos, 0, 1), (qnp.tan, jnp.pi / 4, 1)],
    )
    def test_trig_known_value_and_type(
        self, fn: object, angle_val: float, expected: float
    ) -> None:
        """Each trig function returns dimensionless Quantity with the correct value."""
        result = fn(cxa.Angle(angle_val, "rad"))
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, AbstractAngle)
        assert jnp.allclose(result.value, expected, atol=1e-5)

    @given(angle=_angle_rad)
    def test_pythagorean_identity(self, angle: cxa.Angle) -> None:
        """sin²(x) + cos²(x) == 1 for all angles (numerical sanity check)."""
        s2 = qnp.sin(angle).value ** 2
        c2 = qnp.cos(angle).value ** 2
        assert jnp.allclose(s2 + c2, 1, atol=1e-5)


class TestQuaxedBinary:
    """quaxed.numpy binary ops on Angle produce correct types and values."""

    @pytest.mark.parametrize(
        ("a_val", "b_val", "fn", "expected"),
        [(1, 0.5, qnp.add, 1.5), (1.5, 0.5, qnp.subtract, 1)],
    )
    def test_known_value(
        self, a_val: float, b_val: float, fn: object, expected: float
    ) -> None:
        result = fn(cxa.Angle(a_val, "rad"), cxa.Angle(b_val, "rad"))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, expected)

    def test_multiply_by_scalar(self) -> None:
        """Multiplying an Angle by a plain scalar returns an Angle."""
        result = qnp.multiply(cxa.Angle(1.5, "rad"), 2)
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, 3)

    @given(
        a=cxst.angles(
            unit="rad", elements=st.floats(min_value=0, max_value=3, width=32)
        ),
        b=cxst.angles(
            unit="rad", elements=st.floats(min_value=0, max_value=3, width=32)
        ),
    )
    def test_add_commutativity(self, a: cxa.Angle, b: cxa.Angle) -> None:
        """qnp.add(a, b) == qnp.add(b, a)."""
        assert jnp.allclose(qnp.add(a, b).value, qnp.add(b, a).value, atol=1e-5)

    @given(angle=cxst.angles())
    def test_add_preserves_unit(self, angle: cxa.Angle) -> None:
        assert qnp.add(angle, angle).unit == angle.unit


class TestQuaxedReductions:
    """quaxed.numpy reductions over Angle arrays return Angle with correct values."""

    @pytest.mark.parametrize(
        ("values", "fn", "expected"),
        [
            ([1, 2, 3], qnp.sum, 6),
            ([0, 2, 4], qnp.mean, 2),
            ([3, 1, 2], qnp.min, 1),
            ([3, 1, 2], qnp.max, 3),
        ],
    )
    def test_known_value(
        self, values: list[float], fn: object, expected: float
    ) -> None:
        result = fn(cxa.Angle(values, "rad"))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, expected)

    @pytest.mark.parametrize("fn", [qnp.sum, qnp.mean])
    @given(angle=cxst.angles(shape=(4,)))
    def test_preserves_unit(self, fn: object, angle: cxa.Angle) -> None:
        """Reductions preserve the Angle unit."""
        assert fn(angle).unit == angle.unit


class TestQuaxedArrayOps:
    """quaxed.numpy array manipulation ops preserve Angle type and shape."""

    def test_stack(self) -> None:
        """Stack creates an Angle array from scalar Angles."""
        result = qnp.stack([cxa.Angle(1, "rad"), cxa.Angle(2, "rad")])
        assert isinstance(result, cxa.Angle)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, jnp.array([1, 2]))

    def test_concatenate(self) -> None:
        result = qnp.concatenate([cxa.Angle([1, 2], "rad"), cxa.Angle([3, 4], "rad")])
        assert isinstance(result, cxa.Angle)
        assert result.shape == (4,)
        assert jnp.allclose(result.value, jnp.array([1, 2, 3, 4]))

    def test_sort(self) -> None:
        result = qnp.sort(cxa.Angle([3, 1, 2], "rad"))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, jnp.array([1, 2, 3]))

    def test_reshape(self) -> None:
        result = qnp.reshape(cxa.Angle([[1, 2], [3, 4]], "deg"), (4,))
        assert isinstance(result, cxa.Angle)
        assert result.shape == (4,)

    def test_broadcast_to(self) -> None:
        result = qnp.broadcast_to(cxa.Angle(1.5, "rad"), (3,))
        assert isinstance(result, cxa.Angle)
        assert result.shape == (3,)
        assert jnp.all(result.value == 1.5)

    def test_diff(self) -> None:
        result = qnp.diff(cxa.Angle([1, 3, 6], "deg"))
        assert isinstance(result, cxa.Angle)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, jnp.array([2, 3]))

    @pytest.mark.parametrize(
        ("cond", "a_val", "b_val", "expected"),
        [
            (True, 1.5, 0.5, 1.5),  # True  → picks a
            (False, 1.5, 0.5, 0.5),  # False → picks b
        ],
    )
    def test_where_scalar_cond(
        self, cond: bool, a_val: float, b_val: float, expected: float
    ) -> None:
        """Where with a scalar condition selects the correct Angle."""
        result = qnp.where(
            jnp.array(cond), cxa.Angle(a_val, "rad"), cxa.Angle(b_val, "rad")
        )
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, expected)

    def test_where_array_cond(self) -> None:
        """Where with a boolean array selects element-wise."""
        result = qnp.where(
            jnp.array([True, False, True]),
            cxa.Angle([1, 2, 3], "rad"),
            cxa.Angle([4, 5, 6], "rad"),
        )
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, jnp.array([1, 5, 3]))


class TestQuaxQuaxify:
    """quax.quaxify is the bridge between raw jax.numpy and Angle-aware dispatch.

    Without quaxify, JAX does not know how to handle Angle values and raises
    TypeError. quaxify routes each primitive through the quax dispatch table,
    which has registered rules for AbstractQuantity (and therefore Angle).
    """

    def test_raw_jax_sin_raises_without_quaxify(self) -> None:
        """Raw jax.numpy.sin raises TypeError on an Angle — no dispatch rules."""
        with pytest.raises(TypeError):
            jax.numpy.sin(cxa.Angle(1.5, "rad"))

    @pytest.mark.parametrize(
        ("fn", "angle_val", "expected"),
        [(jax.numpy.sin, jnp.pi / 2, 1), (jax.numpy.cos, 0, 1)],
    )
    def test_quaxify_trig_known_value(
        self, fn: object, angle_val: float, expected: float
    ) -> None:
        """Quaxify wraps a raw jax.numpy trig fn to accept Angles."""
        result = quax.quaxify(fn)(cxa.Angle(angle_val, "rad"))
        assert isinstance(result, u.AbstractQuantity)
        assert jnp.allclose(result.value, expected, atol=1e-6)

    def test_quaxify_add_preserves_angle_type(self) -> None:
        """quaxify(jax.numpy.add) on two Angles returns an Angle."""
        result = quax.quaxify(jax.numpy.add)(cxa.Angle(1, "rad"), cxa.Angle(0.5, "rad"))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, 1.5)

    def test_quaxify_user_function(self) -> None:
        """Quaxify works on a user-defined function that calls raw jax.numpy."""

        def double(x):
            return jax.numpy.add(x, x)

        result = quax.quaxify(double)(cxa.Angle(1, "rad"))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, 2)

    def test_quaxify_jit(self) -> None:
        """jax.jit(quaxify(fn)) works on an Angle."""
        result = jax.jit(quax.quaxify(jax.numpy.sin))(cxa.Angle(jnp.pi / 2, "rad"))
        assert isinstance(result, u.AbstractQuantity)
        assert jnp.allclose(result.value, 1, atol=1e-6)

    def test_quaxify_vmap(self) -> None:
        """jax.vmap(quaxify(fn)) maps over an Angle array."""
        arr = cxa.Angle([0, jnp.pi / 6, jnp.pi / 2], "rad")
        result = jax.vmap(quax.quaxify(jax.numpy.sin))(arr)
        assert isinstance(result, u.AbstractQuantity)
        assert result.value.shape == (3,)
        assert jnp.allclose(result.value[2], 1, atol=1e-5)

    @given(angle=_angle_rad)
    def test_quaxify_pythagorean_identity(self, angle: cxa.Angle) -> None:
        """sin²+cos² == 1 via quaxified raw jax.numpy functions."""
        s = quax.quaxify(jax.numpy.sin)(angle)
        c = quax.quaxify(jax.numpy.cos)(angle)
        assert jnp.allclose(s.value**2 + c.value**2, 1, atol=1e-5)


class TestJAXTransformsWithQuaxed:
    """JAX transforms compose with quaxed.numpy functions on Angle."""

    @pytest.mark.parametrize(
        ("loss_fn", "angle_val", "expected_grad"),
        [
            (lambda x: qnp.sin(x).value, 1.0, jnp.cos(1.0)),  # d/dx sin(x) = cos(x)
            (lambda x: qnp.cos(x).value, 1.0, -jnp.sin(1.0)),  # d/dx cos(x) = -sin(x)
        ],
    )
    def test_grad_known_value(
        self, loss_fn: object, angle_val: float, expected_grad: float
    ) -> None:
        """jax.grad of a scalar qnp function w.r.t. Angle gives the correct grad."""
        g = jax.grad(loss_fn)(cxa.Angle(angle_val, "rad"))
        assert isinstance(g, cxa.Angle)
        assert jnp.allclose(g.value, expected_grad, atol=1e-5)

    @given(angle=_angle_rad_trig)
    @settings(deadline=None)
    def test_grad_sin_equals_cos(self, angle: cxa.Angle) -> None:
        """d/dx sin(x) == cos(x) holds for arbitrary angles in [-1, 1] rad."""
        g = jax.grad(lambda x: qnp.sin(x).value)(angle)
        assert isinstance(g, cxa.Angle)
        assert jnp.allclose(g.value, jnp.cos(angle.value), atol=1e-5)

    @pytest.mark.parametrize(
        ("fn", "angle_val", "expected_type", "expected_val"),
        [
            (qnp.sin, jnp.pi / 2, u.AbstractQuantity, 1),
            (lambda x: qnp.add(x, x), 1, cxa.Angle, 2),
        ],
    )
    def test_jit(
        self, fn: object, angle_val: float, expected_type: type, expected_val: float
    ) -> None:
        """jax.jit works on functions using qnp over Angle."""
        result = jax.jit(fn)(cxa.Angle(angle_val, "rad"))
        assert isinstance(result, expected_type)
        assert jnp.allclose(result.value, expected_val, atol=1e-6)

    @given(angle=cxst.angles(shape=(3,)))
    @settings(deadline=None)
    def test_vmap_abs(self, angle: cxa.Angle) -> None:
        """jax.vmap over qnp.abs maps element-wise and returns an Angle."""
        result = jax.vmap(qnp.abs)(angle)
        assert isinstance(result, cxa.Angle)
        assert result.shape == (3,)
        assert jnp.all(result.value >= 0)

    @given(
        angle=cxst.angles(
            unit="rad",
            shape=(4,),
            elements=st.floats(min_value=-2, max_value=2, width=32),
        )
    )
    @settings(deadline=None)
    def test_vmap_sin(self, angle: cxa.Angle) -> None:
        """jax.vmap over qnp.sin returns a dimensionless Quantity array."""
        result = jax.vmap(qnp.sin)(angle)
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, AbstractAngle)
        assert result.shape == (4,)
