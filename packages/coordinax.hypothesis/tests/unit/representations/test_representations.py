"""Tests for representation strategies."""

import hypothesis.strategies as st
import pytest
from hypothesis import given

import coordinax.representations as cxr

import coordinax.hypothesis.main as cxst
import coordinax.hypothesis.representations as cxsr
from coordinax.hypothesis.utils import get_all_subclasses

# ============================================================================
# valid_basis_classes_for_geometry
# ============================================================================


def test_valid_basis_classes_for_point_geometry() -> None:
    """Point geometry allows only NoBasis."""
    classes = cxst.valid_basis_classes_for_geometry(cxr.PointGeometry())
    assert classes == (cxr.NoBasis,)


def test_valid_basis_classes_for_generic_geometry_returns_all_concrete() -> None:
    """Generic AbstractGeometry fallback returns all concrete basis classes."""
    all_bases = get_all_subclasses(cxr.AbstractBasis, exclude_abstract=True)

    class _AnyGeom(cxr.AbstractGeometry):
        pass

    classes = cxst.valid_basis_classes_for_geometry(_AnyGeom())
    assert set(classes) == set(all_bases)


# ============================================================================
# valid_semantic_classes_for_geometry
# ============================================================================


def test_valid_semantic_classes_for_point_geometry() -> None:
    """Point geometry allows only Location."""
    classes = cxst.valid_semantic_classes_for_geometry(cxr.PointGeometry())
    assert classes == (cxr.Location,)


def test_valid_semantic_classes_for_generic_geometry_returns_all_concrete() -> None:
    """Generic AbstractGeometry fallback returns all concrete semantic classes."""
    all_semantics = get_all_subclasses(cxr.AbstractSemanticKind, exclude_abstract=True)

    class _AnyGeom(cxr.AbstractGeometry):
        pass

    classes = cxst.valid_semantic_classes_for_geometry(_AnyGeom())
    assert set(classes) == set(all_semantics)


# ============================================================================
# representations
# ============================================================================


class TestRepresentations:
    """Tests for the representations strategy."""

    @given(rep=cxsr.representations())
    def test_returns_representation_instance(self, rep: cxr.Representation) -> None:
        """Generated value is a Representation instance."""
        assert isinstance(rep, cxr.Representation)

    @given(rep=cxsr.representations())
    def test_has_correct_fields(self, rep: cxr.Representation) -> None:
        """Generated Representation has geom_kind, basis, and semantic_kind fields."""
        assert isinstance(rep.geom_kind, cxr.AbstractGeometry)
        assert isinstance(rep.basis, cxr.AbstractBasis)
        assert isinstance(rep.semantic_kind, cxr.AbstractSemanticKind)

    @given(rep=cxsr.representations(geom_kind=cxr.PointGeometry()))
    def test_explicit_geom_kind_is_preserved(self, rep: cxr.Representation) -> None:
        """Explicitly provided geom_kind is used."""
        assert isinstance(rep.geom_kind, cxr.PointGeometry)

    @given(rep=cxsr.representations(basis_kind=cxr.NoBasis()))
    def test_explicit_basis_kind_is_preserved(self, rep: cxr.Representation) -> None:
        """Explicitly provided basis_kind is used."""
        assert isinstance(rep.basis, cxr.NoBasis)

    @given(rep=cxsr.representations(semantic_kind=cxr.Location()))
    def test_explicit_semantic_kind_is_preserved(self, rep: cxr.Representation) -> None:
        """Explicitly provided semantic_kind is used."""
        assert isinstance(rep.semantic_kind, cxr.Location)

    @given(
        rep=cxsr.representations(
            geom_kind=cxr.PointGeometry(),
            basis_kind=cxr.NoBasis(),
            semantic_kind=cxr.Location(),
        )
    )
    def test_all_three_explicit_fields(self, rep: cxr.Representation) -> None:
        """All three fields can be specified at once."""
        assert isinstance(rep.geom_kind, cxr.PointGeometry)
        assert isinstance(rep.basis, cxr.NoBasis)
        assert isinstance(rep.semantic_kind, cxr.Location)

    @given(rep=cxsr.representations(geom_kind=cxr.PointGeometry()))
    def test_point_geometry_auto_restricts_basis_and_semantic(
        self, rep: cxr.Representation
    ) -> None:
        """PointGeometry with check_valid=True auto-restricts basis and semantics."""
        assert isinstance(rep.basis, cxr.NoBasis)
        assert isinstance(rep.semantic_kind, cxr.Location)

    @given(
        rep=cxsr.representations(
            geom_kind=st.just(cxr.PointGeometry()),
        )
    )
    def test_strategy_valued_geom_kind_is_drawn(self, rep: cxr.Representation) -> None:
        """Strategy-valued geom_kind is drawn before use."""
        assert isinstance(rep.geom_kind, cxr.PointGeometry)

    @given(
        rep=cxsr.representations(
            basis_kind=st.just(cxr.NoBasis()),
        )
    )
    def test_strategy_valued_basis_kind_is_drawn(self, rep: cxr.Representation) -> None:
        """Strategy-valued basis_kind is drawn before use."""
        assert isinstance(rep.basis, cxr.NoBasis)

    @given(
        rep=cxsr.representations(
            semantic_kind=st.just(cxr.Location()),
        )
    )
    def test_strategy_valued_semantic_kind_is_drawn(
        self, rep: cxr.Representation
    ) -> None:
        """Strategy-valued semantic_kind is drawn before use."""
        assert isinstance(rep.semantic_kind, cxr.Location)

    @given(data=st.data())
    def test_invalid_basis_with_check_valid_raises(self, data: st.DataObject) -> None:
        """Incompatible basis_kind with check_valid=True raises ValueError."""

        class _FakeBasis(cxr.AbstractBasis):
            pass

        with pytest.raises(ValueError, match="Invalid basis_kind"):
            data.draw(
                cxsr.representations(
                    geom_kind=cxr.PointGeometry(),
                    basis_kind=_FakeBasis(),
                    check_valid=True,
                )
            )

    @given(data=st.data())
    def test_invalid_semantic_with_check_valid_raises(
        self, data: st.DataObject
    ) -> None:
        """Incompatible semantic_kind with check_valid=True raises ValueError."""

        class _FakeSemantic(cxr.AbstractSemanticKind):
            @classmethod
            def coord_dimensions(cls, chart, /):
                return tuple(None for _ in chart.components)

        with pytest.raises(ValueError, match="Invalid semantic_kind"):
            data.draw(
                cxsr.representations(
                    geom_kind=cxr.PointGeometry(),
                    semantic_kind=_FakeSemantic(),
                    check_valid=True,
                )
            )

    @given(
        rep=cxsr.representations(
            geom_kind=cxr.PointGeometry(),
            check_valid=False,
        )
    )
    def test_check_valid_false_allows_any_combination(
        self, rep: cxr.Representation
    ) -> None:
        """check_valid=False allows basis/semantic not normally compatible with geom."""
        # With check_valid=False the strategy draws from all available kinds,
        # so we just check the result has the expected geom.
        assert isinstance(rep.geom_kind, cxr.PointGeometry)

    def test_also_accessible_via_main(self) -> None:
        """representations is re-exported from coordinax.hypothesis.main."""
        assert cxst.representations is cxsr.representations
