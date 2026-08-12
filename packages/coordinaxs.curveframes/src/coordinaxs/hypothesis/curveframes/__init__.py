"""Hypothesis strategies for `coordinaxs.curveframes`.

Not imported by `coordinaxs.curveframes` itself -- it is a test/dev-only
extra (`coordinaxs.curveframes[hypothesis]`), imported either directly or,
during a normal test run, as a side effect of `coordinaxs.hypothesis.charts`
loading the `coordinaxs.hypothesis` entry-point group (see
`coordinaxs_hypothesis_exports` below and `pyproject.toml`'s
``[project.entry-points."coordinaxs.hypothesis"]``).
"""

__all__: tuple[str, ...] = ()

from . import _tubular  # noqa: F401


def coordinaxs_hypothesis_exports() -> dict[str, object]:
    """Entry-point provider for the `coordinaxs.hypothesis` group.

    `importlib.metadata.EntryPoint.load()` imports this module before
    fetching the attribute, which is what actually registers the `charts()`
    and `chart_init_kwargs()` overloads for `TubularChart` (defined in
    `._tubular`, imported above). There is nothing this package needs to
    export by name into `coordinaxs.hypothesis`'s own namespace -- the
    strategies are consumed through `charts()`/`chart_init_kwargs()`, so this
    returns no exports.
    """
    return {}
