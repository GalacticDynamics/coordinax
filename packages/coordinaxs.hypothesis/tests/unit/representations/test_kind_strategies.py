"""The `<kind>_classes` / `<kind>s` strategy pairs.

Bases, geometries and semantic kinds are three instances of one strategy shape:
a `*_classes` strategy drawing concrete subclasses of some abstract base, and a
`*s` strategy drawing instances of those classes. They are tested here as that
one shape, parametrized over the three kinds, because everything the old
per-kind modules asserted was identical apart from the base class, one sample
concrete class, and the "no candidates left" error message.
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import hypothesis.strategies as st
import pytest
from hypothesis import given

import coordinax.representations as cxr

import coordinaxs.hypothesis.representations as cxrst
from coordinaxs.hypothesis.utils import get_all_subclasses

KINDS = {
    "basis": SimpleNamespace(
        base=cxr.AbstractBasis,
        classes=cxrst.basis_classes,
        instances=cxrst.bases,
        sample=cxr.NoBasis,
        exhausted_match="No basis classes left after exclusions",
    ),
    "geometry": SimpleNamespace(
        base=cxr.AbstractGeometry,
        classes=cxrst.geometry_classes,
        instances=cxrst.geometries,
        sample=cxr.PointGeometry,
        exhausted_match="No geometry classes left after exclusions",
    ),
    "semantic": SimpleNamespace(
        base=cxr.AbstractSemanticKind,
        classes=cxrst.semantic_classes,
        instances=cxrst.semantics,
        sample=cxr.Location,
        exhausted_match="No semantic classes left after exclusions",
    ),
}


@pytest.fixture(params=sorted(KINDS), scope="module")
def kind(request: pytest.FixtureRequest) -> SimpleNamespace:
    """Each of the three representation-kind strategy pairs.

    Module-scoped: the specs are immutable, and Hypothesis health-checks
    function-scoped fixtures used under `@given` (they would be rebuilt once
    per generated example).
    """
    return KINDS[request.param]


class TestClassStrategies:
    """`<kind>_classes()` draws concrete subclasses of the abstract base."""

    @given(data=st.data())
    def test_returns_subclass_of_base(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert issubclass(data.draw(kind.classes()), kind.base)

    @given(data=st.data())
    def test_never_returns_the_abstract_base(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert data.draw(kind.classes()) is not kind.base

    @given(data=st.data())
    def test_is_concrete_and_instantiable(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """Every drawn class takes no constructor arguments."""
        cls = data.draw(kind.classes())
        assert isinstance(cls(), kind.base)

    @given(data=st.data())
    def test_include_restricts_to_provided_classes(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert data.draw(kind.classes(include=(kind.sample,))) is kind.sample

    @given(data=st.data())
    def test_excluding_everything_raises(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        """Excluding every candidate is an error, not an empty draw."""
        everything = get_all_subclasses(kind.base, exclude_abstract=True)
        with pytest.raises(ValueError, match=kind.exhausted_match):
            data.draw(kind.classes(exclude=tuple(everything)))


class TestInstanceStrategies:
    """`<kind>s()` draws instances of those concrete classes."""

    @given(data=st.data())
    def test_returns_instance_of_base(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert isinstance(data.draw(kind.instances()), kind.base)

    @given(data=st.data())
    def test_never_returns_an_abstract_base_instance(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert type(data.draw(kind.instances())) is not kind.base

    @given(data=st.data())
    def test_include_restricts_to_provided_classes(
        self, kind: SimpleNamespace, data: st.DataObject
    ) -> None:
        assert isinstance(
            data.draw(kind.instances(include=(kind.sample,))), kind.sample
        )
