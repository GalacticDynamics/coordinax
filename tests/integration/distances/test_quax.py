"""Integration tests for coordinax.distances.Distance with quax and quaxed.

Key behavioral contracts verified here:

* ``quaxed.numpy`` arithmetic/unary ops (add, sub, mul, abs, floor, …) that
  preserve non-negativity return ``Distance``.
* ``quaxed.numpy`` negation (``qnp.negative``) returns a plain length
  ``Quantity``, **not** a ``Distance`` — because a negative distance is
  not representable.  This is the critical difference from ``Angle``.
* ``quax.quaxify`` is required to use raw ``jax.numpy`` functions on a
  ``Distance``; without it, JAX raises ``TypeError``.
* JAX transforms (``jit``, ``vmap``, ``grad``) compose correctly with
  ``quaxed.numpy`` operations on ``Distance``.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import pytest
import quax
from hypothesis import given, strategies as st

import quaxed.numpy as qnp
import unxt as u

import coordinax.distances as cxd
import coordinaxs.hypothesis.main as cxst

# ---------------------------------------------------------------------------
# Reusable hypothesis strategies
# ---------------------------------------------------------------------------

# Non-negative, bounded to avoid overflow
_dist_kpc = cxst.distances(
    unit="kpc", elements=st.floats(min_value=0, max_value=3, width=32)
)

# Wider range for vmap/grad tests
_dist_kpc_arr = cxst.distances(
    unit="kpc", shape=(4,), elements=st.floats(min_value=0, max_value=2, width=32)
)


class TestQuaxedUnary:
    """quaxed.numpy unary ops on Distance: most preserve type; negation degrades."""

    # --- correctness on known values ---

    @pytest.mark.parametrize(
        ("value", "fn", "expected"),
        [
            (1.5, qnp.abs, 1.5),
            (1.7, qnp.floor, 1),
            (1.2, qnp.ceil, 2),
            (1.6, qnp.round, 2),
        ],
    )
    def test_known_value_returns_distance(
        self, value: float, fn: object, expected: float
    ) -> None:
        """abs, floor, ceil, round preserve Distance type and return correct value."""
        result = fn(cxd.Distance(value, "kpc"))
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, expected)

    def test_negative_degrades_to_quantity(self) -> None:
        """qnp.negative on a Distance returns a length Quantity, not a Distance.

        Unlike ``Angle`` (where ``-angle`` stays an ``Angle``), a ``Distance``
        is semantically non-negative.  Negating one produces a plain length
        ``Quantity`` so the non-negativity invariant is not violated.
        """
        result = qnp.negative(cxd.Distance(1.5, "kpc"))
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, -1.5)

    # --- property tests ---

    @pytest.mark.parametrize("fn", [qnp.abs, qnp.floor])
    @given(d=cxst.distances())
    def test_preserves_type_and_unit(self, fn: object, d: cxd.Distance) -> None:
        """Abs and floor always return a Distance with the original unit."""
        result = fn(d)
        assert isinstance(result, cxd.Distance)
        assert result.unit == d.unit

    @given(d=cxst.distances())
    def test_abs_nonneg(self, d: cxd.Distance) -> None:
        """qnp.abs of a Distance is always non-negative."""
        assert jnp.all(qnp.abs(d).value >= 0)

    @given(d=cxst.distances())
    def test_negative_is_not_distance(self, d: cxd.Distance) -> None:
        """qnp.negative always returns a non-Distance for any Distance input."""
        assert not isinstance(qnp.negative(d), cxd.Distance)


class TestQuaxedBinary:
    """quaxed.numpy binary ops on Distance produce correct types and values."""

    def test_add_stays_a_distance(self) -> None:
        """Addition is closed: both operands are non-negative, so the sum is."""
        result = qnp.add(cxd.Distance(1, "kpc"), cxd.Distance(0.5, "kpc"))
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 1.5)

    def test_subtract_widens_to_quantity(self) -> None:
        """Subtraction is not closed, so it widens regardless of the values.

        It would be wrong for this to depend on which way round the pair
        falls -- that is precisely the data-dependence the closure policy
        removes.
        """
        result = qnp.subtract(cxd.Distance(1.5, "kpc"), cxd.Distance(0.5, "kpc"))
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 1)

    def test_multiply_by_scalar_widens_to_quantity(self) -> None:
        """A scalar's sign is not knowable at trace time, so scaling widens."""
        result = qnp.multiply(cxd.Distance(1.5, "kpc"), 2)
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 3)

    @given(
        a=cxst.distances(
            unit="kpc", elements=st.floats(min_value=0, max_value=3, width=32)
        ),
        b=cxst.distances(
            unit="kpc", elements=st.floats(min_value=0, max_value=3, width=32)
        ),
    )
    def test_add_commutativity(self, a: cxd.Distance, b: cxd.Distance) -> None:
        """qnp.add(a, b) == qnp.add(b, a)."""
        assert jnp.allclose(qnp.add(a, b).value, qnp.add(b, a).value, atol=1e-5)

    @given(d=cxst.distances())
    def test_add_preserves_unit(self, d: cxd.Distance) -> None:
        assert qnp.add(d, d).unit == d.unit


class TestQuaxedReductions:
    """quaxed.numpy reductions over Distance arrays return Distance."""

    @pytest.mark.parametrize(
        ("values", "fn", "expected"),
        [([1, 2, 3], qnp.sum, 6), ([3, 1, 2], qnp.min, 1), ([3, 1, 2], qnp.max, 3)],
    )
    def test_closed_reductions_stay_distances(
        self, values: list[float], fn: object, expected: float
    ) -> None:
        """sum, min and max are built from closed operations."""
        result = fn(cxd.Distance(values, "kpc"))
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, expected)

    def test_mean_widens_to_quantity(self) -> None:
        """`mean` divides by the count, and division widens.

        The mean of non-negative values *is* non-negative, so this is a real
        cost of the policy rather than a gain: the divisor's sign is no more
        knowable than any other, and exempting it would make the return type
        depend on whether the divisor happens to be a literal.
        """
        result = qnp.mean(cxd.Distance([0, 2, 4], "kpc"))
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 2)

    @pytest.mark.parametrize("fn", [qnp.sum, qnp.mean])
    @given(d=cxst.distances(shape=(4,)))
    def test_preserves_unit(self, fn: object, d: cxd.Distance) -> None:
        """Reductions preserve the Distance unit."""
        assert fn(d).unit == d.unit


class TestQuaxedArrayOps:
    """quaxed.numpy array manipulation ops preserve Distance type and shape."""

    def test_stack(self) -> None:
        """Stack creates a Distance array from scalar Distances."""
        result = qnp.stack([cxd.Distance(1, "kpc"), cxd.Distance(2, "kpc")])
        assert isinstance(result, cxd.Distance)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, jnp.array([1, 2]))

    def test_concatenate(self) -> None:
        result = qnp.concatenate(
            [cxd.Distance([1, 2], "kpc"), cxd.Distance([3, 4], "kpc")]
        )
        assert isinstance(result, cxd.Distance)
        assert result.shape == (4,)
        assert jnp.allclose(result.value, jnp.array([1, 2, 3, 4]))

    def test_sort(self) -> None:
        result = qnp.sort(cxd.Distance([3, 1, 2], "kpc"))
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, jnp.array([1, 2, 3]))

    def test_reshape(self) -> None:
        result = qnp.reshape(cxd.Distance([[1, 2], [3, 4]], "kpc"), (4,))
        assert isinstance(result, cxd.Distance)
        assert result.shape == (4,)

    def test_broadcast_to(self) -> None:
        result = qnp.broadcast_to(cxd.Distance(1.5, "kpc"), (3,))
        assert isinstance(result, cxd.Distance)
        assert result.shape == (3,)
        assert jnp.all(result.value == 1.5)

    def test_diff_widens_to_quantity(self) -> None:
        """`diff` is subtraction, and successive differences can be negative."""
        result = qnp.diff(cxd.Distance([1, 3, 6], "kpc"))
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
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
        """Where with a scalar condition selects the correct Distance."""
        result = qnp.where(
            jnp.array(cond), cxd.Distance(a_val, "kpc"), cxd.Distance(b_val, "kpc")
        )
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, expected)

    def test_where_array_cond(self) -> None:
        """Where with a boolean array selects element-wise."""
        result = qnp.where(
            jnp.array([True, False, True]),
            cxd.Distance([1, 2, 3], "kpc"),
            cxd.Distance([4, 5, 6], "kpc"),
        )
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, jnp.array([1, 5, 3]))


class TestQuaxQuaxify:
    """quax.quaxify is the bridge between raw jax.numpy and Distance-aware dispatch.

    Without quaxify, JAX does not know how to handle Distance values and raises
    TypeError.  quaxify routes each primitive through the quax dispatch table,
    which has registered rules for AbstractQuantity (and therefore Distance).
    """

    def test_raw_jax_add_raises_without_quaxify(self) -> None:
        """Raw jax.numpy.add raises TypeError on a Distance — no dispatch rules."""
        with pytest.raises(TypeError):
            jax.numpy.add(cxd.Distance(1, "kpc"), cxd.Distance(0.5, "kpc"))

    @pytest.mark.parametrize(
        ("fn", "a_val", "b_val", "expected", "stays_distance"),
        [
            (jax.numpy.add, 1, 0.5, 1.5, True),
            # subtraction is not closed, so it widens -- see the closure policy
            (jax.numpy.subtract, 1.5, 0.5, 1, False),
        ],
    )
    def test_quaxify_binary_known_value(
        self,
        fn: object,
        a_val: float,
        b_val: float,
        expected: float,
        stays_distance: bool,
    ) -> None:
        """Quaxify wraps raw jax.numpy binary functions to accept Distances."""
        result = quax.quaxify(fn)(
            cxd.Distance(a_val, "kpc"), cxd.Distance(b_val, "kpc")
        )
        assert isinstance(result, u.AbstractQuantity)
        assert isinstance(result, cxd.Distance) is stays_distance
        assert jnp.allclose(result.value, expected)

    def test_quaxify_user_function(self) -> None:
        """Quaxify works on a user-defined function that calls raw jax.numpy."""

        def double(x):
            return jax.numpy.add(x, x)

        result = quax.quaxify(double)(cxd.Distance(1, "kpc"))
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 2)

    def test_quaxify_jit(self) -> None:
        """jax.jit(quaxify(fn)) works on a Distance."""
        result = jax.jit(quax.quaxify(jax.numpy.add))(
            cxd.Distance(1, "kpc"), cxd.Distance(0.5, "kpc")
        )
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 1.5)

    def test_quaxify_vmap(self) -> None:
        """jax.vmap(quaxify(fn)) maps over a Distance array."""
        arr = cxd.Distance([1, 2, 3], "kpc")
        result = jax.vmap(quax.quaxify(jax.numpy.add))(arr, arr)
        assert isinstance(result, cxd.Distance)
        assert result.value.shape == (3,)
        assert jnp.allclose(result.value, jnp.array([2, 4, 6]))

    @given(d=_dist_kpc)
    def test_quaxify_preserves_distance_type(self, d: cxd.Distance) -> None:
        """quaxify(add)(d, d) always returns a Distance."""
        result = quax.quaxify(jax.numpy.add)(d, d)
        assert isinstance(result, cxd.Distance)
        assert result.unit == d.unit


class TestJAXTransformsWithQuaxed:
    """JAX transforms compose with quaxed.numpy functions on Distance."""

    @pytest.mark.parametrize(
        ("loss_fn", "d_val", "expected_grad"),
        [
            # d/dx sum(x) = 1
            (lambda x: qnp.sum(x).value, 1.0, 1),
            # d/dx 2*x = 2 (via x + x)
            (lambda x: qnp.add(x, x).value, 1.0, 2),
        ],
    )
    def test_grad_known_value(
        self, loss_fn: object, d_val: float, expected_grad: float
    ) -> None:
        """jax.grad of a scalar qnp function w.r.t. Distance gives correct grad."""
        g = jax.grad(loss_fn)(cxd.Distance(d_val, "kpc"))
        assert isinstance(g, cxd.Distance)
        assert jnp.allclose(g.value, expected_grad, atol=1e-5)

    @given(d=_dist_kpc)
    def test_grad_sum_is_one(self, d: cxd.Distance) -> None:
        """d/dx sum(x) == 1 for any Distance scalar."""
        g = jax.grad(lambda x: qnp.sum(x).value)(d)
        assert isinstance(g, cxd.Distance)
        assert jnp.allclose(g.value, 1, atol=1e-5)

    @pytest.mark.parametrize(
        ("fn", "d_val", "expected_val", "stays_distance"),
        [
            (lambda x: qnp.add(x, x), 1, 2, True),
            # scaling widens: a scalar's sign is unknowable at trace time
            (lambda x: qnp.multiply(x, 3), 2, 6, False),
        ],
    )
    def test_jit(
        self, fn: object, d_val: float, expected_val: float, stays_distance: bool
    ) -> None:
        """jax.jit works on functions using qnp over Distance."""
        result = jax.jit(fn)(cxd.Distance(d_val, "kpc"))
        assert isinstance(result, u.AbstractQuantity)
        assert isinstance(result, cxd.Distance) is stays_distance
        assert jnp.allclose(result.value, expected_val, atol=1e-6)

    @given(d=cxst.distances(shape=(3,)))
    def test_vmap_abs(self, d: cxd.Distance) -> None:
        """jax.vmap over qnp.abs maps element-wise and returns a Distance."""
        result = jax.vmap(qnp.abs)(d)
        assert isinstance(result, cxd.Distance)
        assert result.shape == (3,)
        assert jnp.all(result.value >= 0)

    @given(d=_dist_kpc_arr)
    def test_vmap_add(self, d: cxd.Distance) -> None:
        """jax.vmap over qnp.add doubles each element of a Distance array."""
        result = jax.vmap(lambda x: qnp.add(x, x))(d)
        assert isinstance(result, cxd.Distance)
        assert result.shape == (4,)
        assert jnp.allclose(result.value, 2 * d.value, atol=1e-5)
