"""Tests for the semantic_classes and semantics strategies."""

import hypothesis.strategies as st
import pytest
from hypothesis import given

import coordinax.representations as cxr

import coordinax.hypothesis.main as cxst
import coordinax.hypothesis.representations as cxsr
from coordinax.hypothesis.utils import get_all_subclasses

# ============================================================================
# semantic_classes
# ============================================================================


class TestSemanticClasses:
    """Tests for the semantic_classes strategy."""

    @given(sem_cls=cxsr.semantic_classes())
    def test_returns_subclass_of_abstract_semantic_kind(
        self, sem_cls: type[cxr.AbstractSemanticKind]
    ) -> None:
        """Generated class is always a subclass of AbstractSemanticKind."""
        assert issubclass(sem_cls, cxr.AbstractSemanticKind)

    @given(sem_cls=cxsr.semantic_classes())
    def test_never_returns_abstract_base(
        self, sem_cls: type[cxr.AbstractSemanticKind]
    ) -> None:
        """Generated class is never AbstractSemanticKind itself."""
        assert sem_cls is not cxr.AbstractSemanticKind

    @given(sem_cls=cxsr.semantic_classes())
    def test_is_concrete_and_instantiable(
        self, sem_cls: type[cxr.AbstractSemanticKind]
    ) -> None:
        """Generated class is concrete and can be instantiated."""
        instance = sem_cls()
        assert isinstance(instance, cxr.AbstractSemanticKind)

    @given(sem_cls=cxsr.semantic_classes(include=(cxr.Location,)))
    def test_include_restricts_to_provided_classes(
        self, sem_cls: type[cxr.AbstractSemanticKind]
    ) -> None:
        """include parameter restricts generation to provided classes."""
        assert sem_cls is cxr.Location

    @given(data=st.data())
    def test_empty_candidates_raises_value_error(self, data: st.DataObject) -> None:
        """Excluding all candidates raises ValueError."""
        all_semantics = get_all_subclasses(
            cxr.AbstractSemanticKind, exclude_abstract=True
        )
        with pytest.raises(
            ValueError, match="No semantic classes left after exclusions"
        ):
            data.draw(cxsr.semantic_classes(exclude=tuple(all_semantics)))

    def test_also_accessible_via_main(self) -> None:
        """semantic_classes is re-exported from coordinax.hypothesis.main."""
        assert cxst.semantic_classes is cxsr.semantic_classes


# ============================================================================
# semantics
# ============================================================================


class TestSemantics:
    """Tests for the semantics strategy."""

    @given(sem=cxsr.semantics())
    def test_returns_abstract_semantic_kind_instance(
        self, sem: cxr.AbstractSemanticKind
    ) -> None:
        """Generated value is an AbstractSemanticKind instance."""
        assert isinstance(sem, cxr.AbstractSemanticKind)

    @given(sem=cxsr.semantics(include=(cxr.Location,)))
    def test_include_restricts_to_location(self, sem: cxr.AbstractSemanticKind) -> None:
        """include parameter restricts instances to the provided classes."""
        assert isinstance(sem, cxr.Location)

    @given(sem=cxsr.semantics())
    def test_never_returns_abstract_class_instance(
        self, sem: cxr.AbstractSemanticKind
    ) -> None:
        """Generated value is never an instance of the abstract base."""
        assert type(sem) is not cxr.AbstractSemanticKind

    def test_also_accessible_via_main(self) -> None:
        """semantics is re-exported from coordinax.hypothesis.main."""
        assert cxst.semantics is cxsr.semantics
