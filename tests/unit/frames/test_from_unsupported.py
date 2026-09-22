"""`AbstractReferenceFrame.from_` with an unsupported argument (gh-968).

The catch-all registered in `coordinax.frames._src.base` has to do two jobs:
say *what* was unsupported, and never shadow a real dispatch. Both are
checked here, and the first is checked in a subprocess as well, because the
symptom it replaces only appeared under runtime typechecking -- which is read
once, at import, so it cannot be toggled inside a running session.
"""

__all__: tuple[str, ...] = ()

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

import coordinax.frames as cxf


def _is_installed(pkg: str) -> bool:
    """`find_spec`, but an absent parent namespace means "not installed".

    `find_spec` *raises* ModuleNotFoundError (rather than returning None) when
    a parent namespace is absent -- e.g. `coordinaxs` itself on a minimal
    install, or `coordinaxs.interop` when only the astro extra is present -- so
    treat that as "not installed" too. Any other missing module is a genuine
    packaging/runtime failure: let it propagate. Same guard as
    `tests/integration/frames/test_interop_import_order.py`.
    """
    try:
        return importlib.util.find_spec(pkg) is not None
    except ModuleNotFoundError as exc:
        if not f"{pkg}.".startswith(f"{exc.name}."):
            raise
        return False


_HAS_INTEROP = all(
    _is_installed(pkg) for pkg in ("coordinaxs.astro", "coordinaxs.interop.astropy")
)

# ============================================================================
# The unsupported input is named


@pytest.mark.parametrize("obj", [1, "not a frame", object(), None], ids=type)
def test_unsupported_input_names_class_and_argument(obj: object) -> None:
    """The error names the target class and the offending argument's type."""
    with pytest.raises(TypeError, match="Cannot construct 'Alice' from"):
        cxf.Alice.from_(obj)

    with pytest.raises(TypeError, match=type(obj).__qualname__):
        cxf.Alice.from_(obj)


def test_the_offending_value_is_shown_unambiguously() -> None:
    """`repr`, not `str`: a string keeps its quotes rather than reading as a name."""
    with pytest.raises(TypeError, match=r"from 'alice', of type 'str'"):
        cxf.Alice.from_("alice")


def test_a_huge_value_does_not_become_a_huge_error() -> None:
    """The quoted value is flattened to one line and capped.

    `repr` of a 1000-element array runs to thousands of characters over dozens
    of lines -- and so does `str`. An error message is not the place for the
    tail of an array.
    """
    obj = np.arange(1000.0).reshape(100, 10)
    with pytest.raises(TypeError) as exc:
        cxf.Alice.from_(obj)

    msg = str(exc.value)
    assert len(repr(obj)) > 500  # the input really is enormous
    assert "\n" not in msg
    assert len(msg) < 300
    assert msg.startswith("Cannot construct 'Alice' from array([[")
    assert "of type 'ndarray'" in msg


# ============================================================================
# ... and under runtime typechecking too

# The child cold-imports astropy, coordinax and JAX; ~2s idle. See
# `tests/integration/frames/test_interop_import_order.py` for the same budget.
_CHILD_TIMEOUT_S = 300

_CHILD = """
import astropy.coordinates as apyc
import coordinaxs.astro as cxastro
import coordinaxs.interop.astropy  # noqa: F401

try:
    cxastro.ICRS.from_(apyc.FK5())
except TypeError as e:
    assert "Cannot construct 'ICRS' from" in str(e), e
    assert "FK5" in str(e), e
    print("OK")
else:
    raise AssertionError("no error raised")
"""


@pytest.mark.skipif(not _HAS_INTEROP, reason="astropy interop not installed")
@pytest.mark.subprocess_heavy
@pytest.mark.parametrize("typechecking", ["beartype.beartype", ""], ids=["on", "off"])
def test_unsupported_input_is_a_typeerror_either_way(typechecking: str) -> None:
    """Runtime typechecking does not change what an unsupported input raises.

    With it on, plum's own `NotFoundLookupError` could not even be built: the
    first method registered on this function is a jaxtyping wrapper, so
    `plum.Function.owner` looked the owning class up in jaxtyping's module
    namespace and died with ``KeyError('AbstractReferenceFrame')`` -- taking
    with it the message that would have said which argument was unsupported.
    """
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
    if typechecking:
        env["COORDINAX_ENABLE_RUNTIME_TYPECHECKING"] = typechecking
    else:
        env.pop("COORDINAX_ENABLE_RUNTIME_TYPECHECKING", None)

    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        msg = f"child did not finish in {_CHILD_TIMEOUT_S}s; it costs ~2s idle."
        raise AssertionError(msg) from exc

    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout


# ============================================================================
# The catch-all shadows nothing


def test_from_mapping_still_works() -> None:
    """A mapping is more specific than the catch-all."""
    assert cxf.Alice.from_({}) == cxf.Alice()


def test_from_same_frame_still_works() -> None:
    """A frame instance is more specific than the catch-all."""
    assert cxf.AbstractReferenceFrame.from_(cxf.alice) is cxf.alice
    assert cxf.Alice.from_(cxf.alice) is cxf.alice


def test_sibling_frame_keeps_its_own_error() -> None:
    """The frame-instance method still rejects a sibling, with its own message."""
    with pytest.raises(TypeError, match=r"Cannot construct 'Alex' from Alice\(\)"):
        cxf.Alex.from_(cxf.alice)


@pytest.mark.skipif(not _HAS_INTEROP, reason="astropy interop not installed")
class TestAstropyInteropUnshadowed:
    """Every astropy `from_` registration still wins over the catch-all."""

    def test_astropy_frames_convert(self) -> None:
        import astropy.coordinates as apyc

        import coordinaxs.astro as cxastro
        import coordinaxs.interop.astropy  # noqa: F401

        assert cxastro.ICRS.from_(apyc.ICRS()) == cxastro.ICRS()
        assert cxastro.Galactic.from_(apyc.Galactic()) == cxastro.Galactic()
        assert isinstance(
            cxastro.Galactocentric.from_(apyc.Galactocentric()), cxastro.Galactocentric
        )

    def test_astropy_frame_with_data_still_raises_valueerror(self) -> None:
        """The concrete method runs, so its own `ValueError` is not replaced."""
        import astropy.coordinates as apyc
        import astropy.units as apyu

        import coordinaxs.astro as cxastro
        import coordinaxs.interop.astropy  # noqa: F401

        frame = apyc.ICRS(ra=1 * apyu.deg, dec=2 * apyu.deg)
        with pytest.raises(ValueError, match="must not have data"):
            cxastro.ICRS.from_(frame)
