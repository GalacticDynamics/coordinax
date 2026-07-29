"""The contract shared by `coordinax.angles.Angle` and `.distances.Distance`.

Both are `unxt` quantity subclasses that carry a dimension, generate from a
`coordinaxs.hypothesis` strategy, convert between units, do arithmetic, and
survive JAX transforms. Their two test modules asserted all of that twice,
line for line -- both files were exactly 333 lines.

What differs stays per-type, in `angles/test_angle.py` and
`distances/test_distance.py`: angle wrapping, and Distance's non-negativity
(including that `-d` degrades to a plain Quantity, which is the one place the
two types genuinely disagree).
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import plum
import pytest
from hypothesis import given, settings

import quaxed.numpy as qnp
import unxt as u

import coordinax.angles as cxa
import coordinax.distances as cxd
import coordinaxs.hypothesis.main as cxst

KINDS = {
    "angle": SimpleNamespace(
        cls=cxa.Angle,
        strategy=cxst.angles,
        dimension="angle",
        units=("deg", "rad"),
        other_unit="deg",
        bad_unit="m",
        bad_unit_match="angular dimensions",
        # Angle values may be negative, so the algebraic properties are drawn
        # from a symmetric range.
        elements=st.floats(min_value=-1e10, max_value=1e10, width=32),
        algebra_elements=st.floats(min_value=0, max_value=1, width=32),
    ),
    "distance": SimpleNamespace(
        cls=cxd.Distance,
        strategy=cxst.distances,
        dimension="length",
        units=("kpc", "pc"),
        other_unit="pc",
        bad_unit="rad",
        bad_unit_match="dimensions length",
        elements=st.floats(min_value=0, max_value=1e10, width=32),
        algebra_elements=st.floats(min_value=0, max_value=1, width=32),
    ),
}


@pytest.fixture(params=sorted(KINDS), scope="module")
def kind(request: pytest.FixtureRequest) -> SimpleNamespace:
    """Each quantity kind.

    Module-scoped: Hypothesis health-checks function-scoped fixtures used under
    `@given`, which would otherwise be rebuilt once per generated example.
    """
    return KINDS[request.param]


class TestConstruction:
    """Generated values are well-formed instances of their class."""

    @given(data=st.data())
    def test_is_an_instance(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        assert isinstance(data.draw(kind.strategy()), kind.cls)

    @given(data=st.data())
    def test_has_the_right_dimension(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        value = data.draw(kind.strategy())
        assert u.dimension_of(value) == u.dimension(kind.dimension)

    @given(data=st.data())
    def test_unit_matches_requested(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        unit = data.draw(st.sampled_from(kind.units))
        assert data.draw(kind.strategy(unit=unit)).unit == u.unit(unit)

    @given(data=st.data())
    def test_scalar_by_default(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert data.draw(kind.strategy()).shape == ()

    @pytest.mark.parametrize(("shape", "expected"), [(5, (5,)), ((2, 3), (2, 3))])
    @given(data=st.data())
    def test_shape_matches_requested(
        self,
        kind: SimpleNamespace,
        shape: int | tuple[int, ...],
        expected: tuple[int, ...],
        data: st.DataObject,
    ) -> None:
        assert data.draw(kind.strategy(shape=shape)).shape == expected

    def test_invalid_unit_raises(self, kind: SimpleNamespace) -> None:
        """A unit of the wrong dimension is rejected at construction."""
        with pytest.raises(ValueError, match=kind.bad_unit_match):
            kind.cls(1, kind.bad_unit)


class TestConversion:
    """Unit conversion is lossless up to float precision."""

    @given(data=st.data())
    @settings(deadline=None)
    def test_round_trip_through_another_unit(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        value = data.draw(kind.strategy(unit=kind.units[0], elements=kind.elements))
        there = value.uconvert(kind.other_unit)
        back = there.uconvert(kind.units[0])
        assert jnp.allclose(back.value, value.value, rtol=1e-5)


class TestArithmetic:
    """Additive structure, over the kind's own admissible value range.

    The exact algebraic identities (round-trip, commutativity, associativity)
    draw from `algebra_elements`, a narrow [0, 1] band. float32 addition is not
    associative across wide magnitude ranges -- (1.8 + 32767) - 32767 loses
    every significant digit -- so a wide range would be testing the float
    format, not the type. Both original modules made the same distinction.
    """

    @given(data=st.data())
    @settings(deadline=None)
    def test_add_returns_same_kind(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        value = data.draw(kind.strategy(elements=kind.elements))
        assert isinstance(value + value, kind.cls)

    @given(data=st.data())
    @settings(deadline=None)
    def test_sub_self_is_zero(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy(elements=kind.elements))
        assert jnp.allclose((value - value).value, 0)

    @given(data=st.data())
    @settings(deadline=None)
    def test_scalar_mul_scales_value(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        value = data.draw(kind.strategy(elements=kind.elements))
        assert jnp.allclose((2 * value).value, 2 * value.value)

    @given(data=st.data())
    @settings(deadline=None)
    def test_add_then_sub_roundtrips(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        a = data.draw(kind.strategy(unit=kind.units[0], elements=kind.algebra_elements))
        b = data.draw(kind.strategy(unit=kind.units[0], elements=kind.algebra_elements))
        assert jnp.allclose(((a + b) - b).value, a.value, rtol=1e-4, atol=1e-4)

    @given(data=st.data())
    @settings(deadline=None)
    def test_add_is_commutative(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        a = data.draw(kind.strategy(unit=kind.units[0], elements=kind.algebra_elements))
        b = data.draw(kind.strategy(unit=kind.units[0], elements=kind.algebra_elements))
        assert jnp.allclose((a + b).value, (b + a).value)

    @given(data=st.data())
    @settings(deadline=None)
    def test_add_is_associative(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        draw = lambda: data.draw(
            kind.strategy(unit=kind.units[0], elements=kind.algebra_elements)
        )
        a, b, c = draw(), draw(), draw()
        assert jnp.allclose(((a + b) + c).value, (a + (b + c)).value, rtol=1e-4)


class TestJAX:
    """PyTree, jit, vmap and grad all round-trip the type."""

    @given(data=st.data())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy())
        flat, tree = jax.tree.flatten(value)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(value)
        assert restored.unit == value.unit
        assert jnp.array_equal(restored.value, value.value)

    @given(data=st.data())
    @settings(deadline=None)
    def test_jit_identity(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy())
        result = jax.jit(lambda x: x)(value)
        assert type(result) is type(value)
        assert jnp.array_equal(result.value, value.value)

    @given(data=st.data())
    @settings(deadline=None)
    def test_jit_add(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy(elements=kind.elements))
        result = jax.jit(lambda x: x + x)(value)
        assert jnp.allclose(result.value, 2 * value.value)

    @given(data=st.data())
    @settings(deadline=None)
    def test_vmap(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy(shape=(4,), elements=kind.elements))
        result = jax.vmap(lambda x: x + x)(value)
        assert result.shape == (4,)

    @given(data=st.data())
    @settings(deadline=None)
    def test_grad_through_the_value(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """`grad` of `x -> sum(x**2)` is `2x`, with the type stripped."""
        value = data.draw(
            kind.strategy(
                unit=kind.units[0],
                elements=st.floats(min_value=0.5, max_value=10, width=32),
            )
        )
        grad = jax.grad(lambda x: qnp.sum(x**2))(value.value)
        assert jnp.allclose(grad, 2 * value.value, rtol=1e-4)


class TestPlumPromotion:
    """Both kinds promote to, and convert to, a plain Quantity."""

    @given(data=st.data())
    @settings(deadline=None)
    def test_promotes_with_quantity(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        value = data.draw(kind.strategy())
        q = u.Q(1, kind.units[0])

        promoted_value, promoted_q = plum.promote(value, q)
        assert isinstance(promoted_value, u.Q)
        assert isinstance(promoted_q, u.Q)

        assert isinstance(value * q, u.Q)
        assert isinstance(q * value, u.Q)

    @given(data=st.data())
    @settings(deadline=None)
    def test_converts_to_quantity(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """Conversion is a re-wrap: the unit and value objects are shared."""
        value = data.draw(kind.strategy())
        q = plum.convert(value, u.Q)
        assert isinstance(q, u.Q)
        assert q.unit is value.unit
        assert q.value is value.value


# ===================================================================
# Construction from concrete Python / JAX values
#
# Kept out of the fixture-driven classes above because the literals differ per
# kind; the assertions do not.

CONSTRUCT_CASES = [
    pytest.param("angle", 1, "rad", (), id="angle-scalar"),
    pytest.param("angle", [1, 2, 3], "deg", (3,), id="angle-list"),
    pytest.param("distance", 1, "kpc", (), id="distance-scalar"),
    pytest.param("distance", [1, 2, 3], "pc", (3,), id="distance-list"),
]


@pytest.mark.parametrize(("kind_name", "value", "unit_str", "shape"), CONSTRUCT_CASES)
def test_construct_from_python_value(
    kind_name: str, value: object, unit_str: str, shape: tuple[int, ...]
) -> None:
    """Both kinds accept Python ints and lists."""
    cls = KINDS[kind_name].cls
    built = cls(value, unit_str)
    assert isinstance(built, cls)
    assert built.shape == shape


@pytest.mark.parametrize(
    ("kind_name", "unit_str"),
    [
        pytest.param("angle", "rad", id="angle"),
        pytest.param("distance", "kpc", id="distance"),
    ],
)
def test_construct_from_jnp_array(kind_name: str, unit_str: str) -> None:
    """Both kinds accept a JAX array."""
    cls = KINDS[kind_name].cls
    built = cls(jnp.array([0, 1, 2]), unit_str)
    assert isinstance(built, cls)
    assert built.shape == (3,)
