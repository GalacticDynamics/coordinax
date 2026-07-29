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
from hypothesis import given, settings

import coordinax.distances as cxd
import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro as cxastrost
import coordinaxs.hypothesis.distances as cxdst

KINDS = {
    "distance": SimpleNamespace(cls=cxd.Distance, strategy=cxdst.distances),
    "distance_modulus": SimpleNamespace(
        cls=cxastro.DistanceModulus, strategy=cxastrost.distance_moduli
    ),
    "parallax": SimpleNamespace(cls=cxastro.Parallax, strategy=cxastrost.parallaxes),
}


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
        value = data.draw(kind.strategy())
        result = jax.jit(lambda x: x + x)(value)
        assert jnp.allclose(result.value, 2 * value.value)

    @given(data=st.data())
    @settings(deadline=None)
    def test_vmap(self, kind: SimpleNamespace, data: st.DataObject) -> None:
        value = data.draw(kind.strategy(shape=(3,)))
        result = jax.vmap(lambda x: x + x)(value)
        assert result.shape == (3,)
