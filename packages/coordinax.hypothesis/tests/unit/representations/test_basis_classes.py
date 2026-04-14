"""Tests for the basis_classes and bases strategies."""

import hypothesis.strategies as st
import pytest
from hypothesis import given

import coordinax.representations as cxr

import coordinax.hypothesis.main as cxst
import coordinax.hypothesis.representations as cxsr
from coordinax.hypothesis.utils import get_all_subclasses

# ============================================================================
# basis_classes
# ============================================================================


class TestBasisClasses:
    """Tests for the basis_classes strategy."""

    @given(basis_cls=cxsr.basis_classes())
    def test_returns_subclass_of_abstract_basis(
        self, basis_cls: type[cxr.AbstractBasis]
    ) -> None:
        """Generated class is always a subclass of AbstractBasis."""
        assert issubclass(basis_cls, cxr.AbstractBasis)

    @given(basis_cls=cxsr.basis_classes())
    def test_never_returns_abstract_base(
        self, basis_cls: type[cxr.AbstractBasis]
    ) -> None:
        """Generated class is never AbstractBasis itself."""
        assert basis_cls is not cxr.AbstractBasis

    @given(basis_cls=cxsr.basis_classes())
    def test_is_concrete_and_instantiable(
        self, basis_cls: type[cxr.AbstractBasis]
    ) -> None:
        """Generated class is concrete and can be instantiated."""
        instance = basis_cls()
        assert isinstance(instance, cxr.AbstractBasis)

    @given(basis_cls=cxsr.basis_classes(include=(cxr.NoBasis,)))
    def test_include_restricts_to_provided_classes(
        self, basis_cls: type[cxr.AbstractBasis]
    ) -> None:
        """include parameter restricts generation to provided classes."""
        assert basis_cls is cxr.NoBasis

    @given(data=st.data())
    def test_empty_candidates_raises_value_error(self, data: st.DataObject) -> None:
        """Excluding all candidates raises ValueError."""
        all_bases = get_all_subclasses(cxr.AbstractBasis, exclude_abstract=True)
        with pytest.raises(ValueError, match="No basis classes left after exclusions"):
            data.draw(cxsr.basis_classes(exclude=tuple(all_bases)))

    def test_also_accessible_via_main(self) -> None:
        """basis_classes is re-exported from coordinax.hypothesis.main."""
        assert cxst.basis_classes is cxsr.basis_classes


# ============================================================================
# bases
# ============================================================================


class TestBases:
    """Tests for the bases strategy."""

    @given(basis=cxsr.bases())
    def test_returns_abstract_basis_instance(self, basis: cxr.AbstractBasis) -> None:
        """Generated value is an AbstractBasis instance."""
        assert isinstance(basis, cxr.AbstractBasis)

    @given(basis=cxsr.bases(include=(cxr.NoBasis,)))
    def test_include_restricts_to_no_basis(self, basis: cxr.AbstractBasis) -> None:
        """include parameter restricts instances to the provided classes."""
        assert isinstance(basis, cxr.NoBasis)

    @given(basis=cxsr.bases())
    def test_never_returns_abstract_class_instance(
        self, basis: cxr.AbstractBasis
    ) -> None:
        """Generated value is never an instance of the abstract base."""
        assert type(basis) is not cxr.AbstractBasis

    def test_also_accessible_via_main(self) -> None:
        """bases is re-exported from coordinax.hypothesis.main."""
        assert cxst.bases is cxsr.bases
