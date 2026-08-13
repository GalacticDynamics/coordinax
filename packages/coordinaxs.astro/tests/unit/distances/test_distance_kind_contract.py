"""The contract shared by every distance-like quantity kind.

`Distance`, `Parallax` and `DistanceModulus` are three `AbstractDistance`
subclasses with three strategies of the same shape. Shape handling, `elements`
forwarding, `st.from_type` registration and JAX/PyTree behaviour are identical
across all three, so they are asserted once here.

What genuinely differs -- Parallax's non-negativity check, DistanceModulus's
fixed 'mag' unit, and the conversions between them -- stays in
`test_parallax.py` / `test_distance_modulus.py`.
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given

import unxt as u

import coordinax.distances as cxd
import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro as cxastrost
import coordinaxs.hypothesis.distances as cxdst

#: `dimension` is what both the class and its instances must report.
#: `sign_constrained` records whether the kind's domain excludes negatives.
#: `Distance` and `Parallax` are non-negative by construction, so negating one
#: cannot yield a value of the same type. `DistanceModulus` is two-sided --
#: dm = 5 log10(d/10pc) maps d in (0, inf) onto all of the reals, and a
#: negative dm is the ordinary way to say "nearer than 10 pc".
KINDS = {
    "distance": SimpleNamespace(
        cls=cxd.Distance,
        strategy=cxdst.distances,
        dimension=u.dimension("length"),
        sign_constrained=True,
    ),
    "distance_modulus": SimpleNamespace(
        cls=cxastro.DistanceModulus,
        strategy=cxastrost.distance_moduli,
        dimension=u.dimension_of(u.Q(1.0, "mag")),
        sign_constrained=False,
    ),
    "parallax": SimpleNamespace(
        cls=cxastro.Parallax,
        strategy=cxastrost.parallaxes,
        dimension=u.dimension("angle"),
        sign_constrained=True,
    ),
}


def _a_valid_unit(kind: SimpleNamespace) -> str:
    """A unit the kind's `__check_init__` accepts."""
    return {
        cxd.Distance: "kpc",
        cxastro.Parallax: "mas",
        cxastro.DistanceModulus: "mag",
    }[kind.cls]


@pytest.fixture(params=sorted(KINDS), scope="module")
def kind(request: pytest.FixtureRequest) -> SimpleNamespace:
    """Each distance-like quantity kind.

    Module-scoped: Hypothesis health-checks function-scoped fixtures used
    under `@given`, which would be rebuilt once per generated example.
    """
    return KINDS[request.param]


class TestGeneration:
    """The strategy honours shape and element constraints."""

    @given(data=st.data())
    def test_is_an_instance_of_its_class(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert isinstance(data.draw(kind.strategy()), kind.cls)

    @given(data=st.data())
    def test_is_an_abstract_distance(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert isinstance(data.draw(kind.strategy()), cxd.AbstractDistance)

    @given(data=st.data())
    def test_scalar_by_default(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert data.draw(kind.strategy()).shape == ()

    @pytest.mark.parametrize(
        ("shape", "expected"), [(5, (5,)), ((3,), (3,)), ((2, 3), (2, 3))]
    )
    @given(data=st.data())
    def test_shape_is_honoured(
        self,
        kind: SimpleNamespace,
        shape: int | tuple[int, ...],
        expected: tuple[int, ...],
        data: st.DataObject,
    ) -> None:
        assert data.draw(kind.strategy(shape=shape)).shape == expected

    @given(data=st.data())
    def test_elements_bounds_are_honoured(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """`elements=` narrows the generated values."""
        value = data.draw(
            kind.strategy(elements=st.floats(min_value=1, max_value=30, width=32))
        )
        assert 1 <= float(value.value) <= 30


class TestFromType:
    """`st.from_type` resolves to the registered strategy."""

    @given(data=st.data())
    def test_from_type_generates_the_class(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert isinstance(data.draw(st.from_type(kind.cls)), kind.cls)

    @given(data=st.data())
    def test_builds_can_use_from_type(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """`st.builds` resolves the annotation through `from_type`."""
        strategy = st.builds(lambda q: float(q.value.item()), q=st.from_type(kind.cls))
        assert isinstance(data.draw(strategy), float)


class TestJAX:
    """PyTree, jit and vmap behaviour is the same for all three."""

    @given(data=st.data())
    def test_pytree_roundtrip(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy())
        flat, tree = jax.tree.flatten(value)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(value)
        assert restored.unit == value.unit
        assert jnp.array_equal(restored.value, value.value)

    @given(data=st.data())
    def test_jit_identity(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy())
        result = jax.jit(lambda x: x)(value)
        assert type(result) is type(value)
        assert jnp.array_equal(result.value, value.value)

    @given(data=st.data())
    def test_jit_add(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy())
        result = jax.jit(lambda x: x + x)(value)
        assert jnp.allclose(result.value, 2 * value.value)

    @given(data=st.data())
    def test_vmap(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy(shape=(3,)))
        result = jax.vmap(lambda x: x + x)(value)
        assert result.shape == (3,)


class TestNegation:
    """Negation degrades exactly for the kinds that constrain sign.

    Regression: the `neg_p` rule was registered on the concrete `Distance`
    rather than on `AbstractDistance`, so `-Parallax(...)` fell through to
    unxt's type-preserving rule, rebuilt a `Parallax` holding a negative value,
    and tripped that class's own `check_negative` guard -- it raised
    `EquinoxRuntimeError` rather than returning anything.
    """

    def test_degrades_iff_sign_constrained(self, kind: SimpleNamespace) -> None:
        """A sign-constrained kind yields `Quantity`; a two-sided one keeps its type."""
        result = -kind.cls(10, _a_valid_unit(kind))

        if kind.sign_constrained:
            assert type(result) is u.Q
        else:
            assert type(result) is kind.cls

    def test_negation_preserves_magnitude_and_unit(self, kind: SimpleNamespace) -> None:
        unit = _a_valid_unit(kind)
        result = -kind.cls(10, unit)
        assert jnp.array_equal(result.value, -10)
        assert result.unit == u.unit(unit)

    def test_survives_jit(self, kind: SimpleNamespace) -> None:
        """`check_negative` is an `error_if`, so jit is where it would fire."""
        result = jax.jit(lambda x: -x)(kind.cls(10, _a_valid_unit(kind)))
        expected = u.Q if kind.sign_constrained else kind.cls
        assert type(result) is expected

    def test_survives_vmap(self, kind: SimpleNamespace) -> None:
        value = kind.cls(jnp.asarray([1.0, 2.0]), _a_valid_unit(kind))
        result = jax.vmap(lambda x: -x)(value)
        assert jnp.array_equal(result.value, jnp.asarray([-1.0, -2.0]))

    def test_two_sided_kinds_negate_back(self, kind: SimpleNamespace) -> None:
        """Where negation is closed it is also an involution."""
        if kind.sign_constrained:
            pytest.skip("negation is not closed on this kind")
        original = kind.cls(-5, _a_valid_unit(kind))
        once = -original
        twice = -once
        assert type(twice) is kind.cls
        assert jnp.array_equal(twice.value, original.value)


class TestDimensionOf:
    """`dimension_of` agrees whether asked about the class or an instance.

    Regression: `coordinax.distances` registers `dimension_of` for
    `type[AbstractDistance]` returning length. That is right for
    `AbstractDistance` and for `Distance`, but it was inherited by `Parallax`
    (angle) and `DistanceModulus` (magnitude), so asking those two classes
    their dimension gave `length` while every instance of them said otherwise.
    """

    def test_class_matches_its_declared_dimension(self, kind: SimpleNamespace) -> None:
        assert u.dimension_of(kind.cls) == kind.dimension

    @given(data=st.data())
    def test_instance_matches_its_class(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """The point of the fix: the two answers cannot disagree."""
        instance = data.draw(kind.strategy())
        assert u.dimension_of(instance) == u.dimension_of(kind.cls)

    def test_the_base_class_still_reports_length(self) -> None:
        """The inherited rule is correct for the abstraction itself; keep it."""
        assert u.dimension_of(cxd.AbstractDistance) == u.dimension("length")


class TestArithmeticClosure:
    """A kind survives an operation exactly when that operation is closed on it.

    The policy: return the kind iff closure is a *theorem*, and widen to
    `Quantity` otherwise -- never decide at runtime. For the sign-constrained
    kinds only addition qualifies; `DistanceModulus` spans the reals, so
    subtraction and scalar scaling are closed on it too.

    The point of asserting this per-kind is that the alternative -- preserving
    the type and validating the result -- makes success depend on the values
    rather than the types, which is unusable under `jit` and `vmap`.
    """

    def _expected(self, kind: SimpleNamespace) -> type:
        return u.quantity.Quantity if kind.sign_constrained else kind.cls

    def test_addition_is_always_closed(self, kind: SimpleNamespace) -> None:
        """Both operands are in the domain, so the sum is too -- every kind."""
        unit = _a_valid_unit(kind)
        result = kind.cls(1, unit) + kind.cls(2, unit)
        assert type(result) is kind.cls

    def test_subtraction_towards_a_negative_result(self, kind: SimpleNamespace) -> None:
        unit = _a_valid_unit(kind)
        result = kind.cls(1, unit) - kind.cls(3, unit)
        assert type(result) is self._expected(kind)
        assert float(u.ustrip(unit, result)) == pytest.approx(-2.0)

    def test_subtraction_towards_a_positive_result_agrees(
        self, kind: SimpleNamespace
    ) -> None:
        """The same op must not change type with the data -- that is the bug."""
        unit = _a_valid_unit(kind)
        negative = kind.cls(1, unit) - kind.cls(3, unit)
        positive = kind.cls(3, unit) - kind.cls(1, unit)
        assert type(negative) is type(positive)

    @pytest.mark.parametrize("scalar", [2, -1, 0])
    def test_scalar_multiplication(self, kind: SimpleNamespace, scalar: int) -> None:
        unit = _a_valid_unit(kind)
        assert type(kind.cls(3, unit) * scalar) is self._expected(kind)
        assert type(scalar * kind.cls(3, unit)) is self._expected(kind)

    @pytest.mark.parametrize("scalar", [2, -2])
    def test_scalar_division(self, kind: SimpleNamespace, scalar: int) -> None:
        unit = _a_valid_unit(kind)
        assert type(kind.cls(6, unit) / scalar) is self._expected(kind)

    def test_no_arithmetic_raises(self, kind: SimpleNamespace) -> None:
        """Totality: none of these may depend on the values to succeed."""
        unit = _a_valid_unit(kind)
        a, b = kind.cls(1, unit), kind.cls(3, unit)
        for op in (
            lambda: a - b,
            lambda: b - a,
            lambda: a * -1,
            lambda: -1 * a,
            lambda: a / -2,
            lambda: a + b,
            lambda: -a,
        ):
            op()  # must not raise

    def test_batched_subtraction_does_not_fail_on_one_element(
        self, kind: SimpleNamespace
    ) -> None:
        """A single negative element used to poison the whole array."""
        unit = _a_valid_unit(kind)
        lhs = kind.cls(jnp.asarray([3.0, 1.0]), unit)
        rhs = kind.cls(jnp.asarray([1.0, 3.0]), unit)
        assert jnp.array_equal(u.ustrip(unit, lhs - rhs), jnp.asarray([2.0, -2.0]))

    def test_under_vmap(self, kind: SimpleNamespace) -> None:
        unit = _a_valid_unit(kind)
        out = jax.vmap(
            lambda a, b: u.ustrip(unit, kind.cls(a, unit) - kind.cls(b, unit))
        )(jnp.asarray([3.0, 1.0]), jnp.asarray([1.0, 3.0]))
        assert jnp.array_equal(out, jnp.asarray([2.0, -2.0]))

    def test_dimension_changing_products_always_widen(
        self, kind: SimpleNamespace
    ) -> None:
        """`x * x` squares the unit, so it is never the same kind."""
        unit = _a_valid_unit(kind)
        result = kind.cls(2, unit) * kind.cls(3, unit)
        assert type(result) is u.quantity.Quantity

    def test_reciprocal_always_widens(self, kind: SimpleNamespace) -> None:
        """`1 / x` inverts the unit, so it is never the same kind."""
        result = 1 / kind.cls(2, _a_valid_unit(kind))
        assert type(result) is u.quantity.Quantity
