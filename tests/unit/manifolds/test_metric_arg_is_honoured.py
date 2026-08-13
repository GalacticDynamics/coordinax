"""Every metric-level dispatch must honour the metric it is handed.

A metric-level overload takes ``(metric, chart, ...)``, but every primitive
underneath -- `quadratic_form`, `gram`, `metric_matrix` -- reads
``chart.M.metric``. So the argument can never be *used*; it selects the method
and must otherwise be checked. Skipping the check is silent: the tests all pass
the matching metric, so the ignored argument never differs from the one
actually applied, and nothing fails.

Hence #674, #680, #695 -- and `angle_between`/`norm` here.
`test_no_unguarded_metric_overloads` reads the source, so a new overload that
forgets is caught without anyone remembering to add a case.
"""

__all__: tuple[str, ...] = ()

import ast
import pathlib

import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

USYS = u.unitsystem("m", "s", "kg", "rad")
Z3 = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
P3 = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
Z4 = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
X4 = {"ct": u.Q(0.0, "m"), "x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
Y4 = {"ct": u.Q(0.0, "m"), "x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")}

#: Metrics no chart below carries: Minkowski against the 3-D charts, flat
#: against the Minkowski one.
MINK = cxm.MinkowskiMetric()
FLAT4 = cxm.FlatMetric(4)

#: A point on `cartnd`, whose manifold `Rn(N)` leaves its dimension open.
QN = {"q": u.Q(jnp.asarray([1.0, 2.0, 3.0]), "m")}


#: ``(id, thunk)`` where the thunk passes a metric the chart does not carry.
#: One row per metric-level overload reachable with a mismatched metric.
MISMATCHED_CALLS = [
    ("norm-cdict", lambda: cxm.norm(P3, MINK, cxc.cart3d, at=Z3)),
    (
        "norm-bare-array",
        lambda: cxm.norm(
            jnp.asarray([1.0, 0.0, 0.0]), MINK, cxc.cart3d, at=Z3, usys=USYS
        ),
    ),
    # `FlatMetric(2)` and `FlatMetric(3)` both satisfy the `FlatMetric`
    # annotation, so the Cartesian short-circuit's dispatch does not pin the
    # dimension and the guard is what catches it.
    (
        "norm-flat-wrong-ndim",
        lambda: cxm.norm(
            jnp.asarray([1.0, 0.0, 0.0]),
            cxm.FlatMetric(2),
            cxc.cart3d,
            at=Z3,
            usys=USYS,
        ),
    ),
    ("separation", lambda: cxm.separation(MINK, cxc.cart3d, Z3, P3)),
    ("angle_between-3d", lambda: cxm.angle_between(MINK, cxc.cart3d, P3, P3, at=Z3)),
    ("scale_factors", lambda: cxm.scale_factors(MINK, cxc.cart3d, at=P3)),
    ("interval", lambda: cxm.interval(FLAT4, cxc.minkowskict, Z4, X4)),
    (
        "angle_between-4d",
        lambda: cxm.angle_between(FLAT4, cxc.minkowskict, X4, Y4, at=Z4),
    ),
    # `Rn(N)` relaxes to a kind check, but a kind check is still a check.
    (
        "scale_factors-open-ndim-wrong-kind",
        lambda: cxm.scale_factors(MINK, cxc.cartnd, at=QN),
    ),
]


@pytest.mark.parametrize(
    ("name", "call"), MISMATCHED_CALLS, ids=[c[0] for c in MISMATCHED_CALLS]
)
def test_mismatched_metric_is_refused(name: str, call) -> None:
    """A metric that is not the chart's must raise, not be quietly dropped."""
    del name
    with pytest.raises(ValueError, match="metric-level dispatch needs the chart's own"):
        call()


def test_open_dimension_chart_accepts_any_metric_of_its_kind() -> None:
    """`Rn(N)` pins no dimension, so the caller's metric supplies it.

    The premise "the metric argument is never data" holds only for a chart
    whose manifold fixes its dimension. `cxc.cartnd` does not, so demanding
    equality against its unbound `FlatMetric(ndim=True)` would reject the
    legitimate call that #708 relies on.
    """
    got = cxm.scale_factors(cxm.FlatMetric(3), cxc.cartnd, at=QN)
    assert got.shape[-1] == 3


def test_matching_metric_still_works() -> None:
    """The guard must not reject the legitimate call."""
    assert (
        float(cxm.norm(P3, cxc.cart3d.M.metric, cxc.cart3d, at=Z3).ustrip("m")) == 1.0
    )
    assert cxm.interval(cxc.minkowskict.M.metric, cxc.minkowskict, Z4, X4) is not None


# ---------------------------------------------------------------------------
# The completeness check: the reason this closes the class and not the cases.

_SRC = pathlib.Path(cxm.__file__).parent.parent / "_src"


def _calls_guard(node: ast.FunctionDef) -> bool:
    """Report whether the body actually *calls* the guard.

    A call node, not a substring of the unparsed source: a docstring or comment
    naming `check_metric_is_charts` must not be able to satisfy this.
    """
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "check_metric_is_charts"
        for n in ast.walk(node)
    )


def _forwards_metric(node: ast.FunctionDef) -> bool:
    """Report whether the body passes its own ``metric`` to another verb.

    Only a dispatched verb counts -- an attribute call such as
    ``cxmapi.interval(metric, ...)``. A bare-name call taking ``metric`` (the
    guard itself, or a local helper) is not forwarding, and a body that merely
    mentions ``cxmapi`` while passing ``chart.M`` is not either. Both are
    exactly the shapes this file exists to catch.
    """
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and any(isinstance(a, ast.Name) and a.id == "metric" for a in n.args)
        for n in ast.walk(node)
    )


def _is_refusal(node: ast.FunctionDef) -> bool:
    """Report whether the body raises instead of ever reaching a metric."""
    return any(
        isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name)
        and n.exc.func.id == "NotImplementedError"
        for n in ast.walk(node)
    )


def _metric_level_overloads() -> list[tuple[str, int, ast.FunctionDef]]:
    """Every ``def f(..., metric, chart, ...)`` under ``_src``, found by source."""
    found = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "check_metric_is_charts":
                continue  # the guard itself, not a dispatch that needs guarding
            names = {a.arg for a in node.args.posonlyargs + node.args.args}
            if {"metric", "chart"} <= names:
                found.append((str(path.relative_to(_SRC)), node.lineno, node))
    return found


def test_no_unguarded_metric_overloads() -> None:
    """No metric-level overload may silently ignore its metric.

    Source-level rather than behavioural on purpose: a new overload added later
    is covered the moment it is written, without anyone remembering to extend
    `WRONG_METRIC_CALLS` above.
    """
    overloads = _metric_level_overloads()
    assert overloads, "introspection found nothing -- the check would vacuously pass"

    unguarded = [
        f"{path}:{line}"
        for path, line, node in overloads
        if not _calls_guard(node)
        and not _forwards_metric(node)
        and not _is_refusal(node)
    ]
    assert not unguarded, (
        "these take a `metric` but neither check it against the chart nor "
        f"forward it to a verb that does: {unguarded}"
    )
