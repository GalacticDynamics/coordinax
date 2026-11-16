"""Guard: the runtime type-checking hook reaches the private modules.

jaxtyping's import hook instruments a module only when the module's own name
matches the hook's prefix at the moment it is first imported. The code lives in
``coordinax._src.*`` and in each workspace package's ``_src``, so a hook on a
public module name (``coordinax.charts``) silently checks nothing.
"""

import os
import subprocess
import sys

import pytest

# One function from each place code lives. Each is a plain module-level
# function, so the hook decorates it and leaves a ``__wrapped__`` behind.
REPRESENTATIVES = (
    "coordinax._src.charts.checks:polar_range",
    "coordinax._src.charts.checks:check_manifolds_match_charts",
    "coordinax.representations._src.semantics:Location.__init__",
    "coordinaxs.astro._src.distance_modulus:_distance_modulus_from_pc",
    "coordinaxs.curveframes._src.base:unit_or_none",
    "coordinaxs.interop.astropy._src.angles:convert_astropy_angle_to_cx_angle",
)

_PROBE = """
import importlib, operator, sys
import coordinax, coordinaxs.astro, coordinaxs.curveframes, coordinaxs.interop.astropy
for spec in sys.argv[1:]:
    mod, _, attr = spec.partition(":")
    obj = operator.attrgetter(attr)(importlib.import_module(mod))
    print(spec, hasattr(obj, "__wrapped__"))
"""


def _wrapped(env_value: str | None) -> dict[str, bool]:
    env = {**os.environ}
    env.pop("COORDINAX_ENABLE_RUNTIME_TYPECHECKING", None)
    if env_value is not None:
        env["COORDINAX_ENABLE_RUNTIME_TYPECHECKING"] = env_value
    out = subprocess.run(  # noqa: S603  # fixed literal script, this interpreter
        [sys.executable, "-c", _PROBE, *REPRESENTATIVES],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {k: v == "True" for k, v in (ln.split() for ln in out.splitlines())}


@pytest.mark.parametrize(
    ("env_value", "expected"), [("beartype.beartype", True), (None, False)]
)
def test_private_modules_are_instrumented(
    env_value: str | None, *, expected: bool
) -> None:
    """With the env var set every representative is wrapped; without, none is."""
    assert _wrapped(env_value) == dict.fromkeys(REPRESENTATIVES, expected)
