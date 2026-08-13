"""Every metric-level dispatch must honour the metric it is handed.

A metric-level overload takes ``(metric, chart, ...)``, but every primitive
underneath -- `quadratic_form`, `gram`, `metric_matrix` -- reads
``chart.M.metric``. So the argument can never be *used*; it selects the method
and must otherwise be checked. Skipping the check is silent: the tests all pass
the matching metric, so the ignored argument never differs from the one
actually applied, and nothing fails.

That is why this defect recurred four times (#674, #680, #695, and again in
`angle_between`/`norm` here). These tests target the *shape* of the mistake
rather than the instances: `test_no_unguarded_metric_overloads` reads the
source, so a newly added overload that forgets is caught without anyone
remembering to add a case for it.
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


def _four_velocity(beta: float) -> dict[str, u.AbstractQuantity]:
    gamma = 1.0 / jnp.sqrt(1.0 - beta**2)
    return {
        "ct": u.Q(gamma, ""),
        "x": u.Q(gamma * beta, ""),
        "y": u.Q(0.0, ""),
        "z": u.Q(0.0, ""),
    }


#: ``(id, call)`` where ``call`` takes the *wrong* metric and must refuse it.
#: One row per metric-level overload reachable with a mismatched metric.
WRONG_METRIC_CALLS = [
    ("norm-cdict", lambda m: cxm.norm(P3, m, cxc.cart3d, at=Z3)),
    (
        "norm-bare-array",
        lambda m: cxm.norm(
            jnp.asarray([1.0, 0.0, 0.0]), m, cxc.cart3d, at=Z3, usys=USYS
        ),
    ),
    ("separation", lambda m: cxm.separation(m, cxc.cart3d, Z3, P3)),
    ("angle_between-3d", lambda m: cxm.angle_between(m, cxc.cart3d, P3, P3, at=Z3)),
    ("interval", lambda m: cxm.interval(m, cxc.minkowskict, Z4, X4)),
    (
        "angle_between-4d",
        lambda m: cxm.angle_between(m, cxc.minkowskict, X4, Y4, at=Z4),
    ),
    ("scale_factors", lambda m: cxm.scale_factors(m, cxc.cart3d, at=P3)),
]

#: A metric that is *not* the chart's, per chart dimension.
WRONG = {3: cxm.MinkowskiMetric(), 4: cxm.FlatMetric(4)}


@pytest.mark.parametrize(
    ("name", "call"), WRONG_METRIC_CALLS, ids=[c[0] for c in WRONG_METRIC_CALLS]
)
def test_mismatched_metric_is_refused(name: str, call) -> None:
    """A metric that is not the chart's must raise, not be quietly dropped."""
    wrong = WRONG[4] if "4d" in name or name == "interval" else WRONG[3]
    with pytest.raises(ValueError, match="metric-level dispatch needs the chart's own"):
        call(wrong)


def test_flat_metric_fast_path_checks_dimension() -> None:
    """The Cartesian short-circuit is dispatch-gated but still not exempt.

    ``FlatMetric(2)`` and ``FlatMetric(3)`` both satisfy the ``FlatMetric``
    annotation, so dispatch alone does not pin the dimension.
    """
    with pytest.raises(ValueError, match="metric-level dispatch needs the chart's own"):
        cxm.norm(
            jnp.asarray([1.0, 0.0, 0.0]),
            cxm.FlatMetric(2),
            cxc.cart3d,
            at=Z3,
            usys=USYS,
        )


def test_matching_metric_still_works() -> None:
    """The guard must not reject the legitimate call."""
    assert (
        float(cxm.norm(P3, cxc.cart3d.M.metric, cxc.cart3d, at=Z3).ustrip("m")) == 1.0
    )
    assert cxm.interval(cxc.minkowskict.M.metric, cxc.minkowskict, Z4, X4) is not None
    phi = cxm.lorentzian.rapidity_between(
        cxc.minkowskict.M.metric,
        cxc.minkowskict,
        _four_velocity(0.0),
        _four_velocity(0.6),
        at=Z4,
    )
    assert float(phi) == pytest.approx(float(jnp.arctanh(0.6)), abs=1e-6)


# ---------------------------------------------------------------------------
# The completeness check: the reason this closes the class and not the cases.

_SRC = pathlib.Path(cxm.__file__).parent.parent / "_src"


def _forwards_metric(node: ast.FunctionDef) -> bool:
    """Report whether the body passes its own ``metric`` to another verb.

    Checked structurally, not by substring: a body that merely mentions
    ``cxmapi`` while forwarding ``chart.M`` is *not* forwarding the metric, and
    that is exactly the bug this file exists to catch.
    """
    return any(
        isinstance(n, ast.Call)
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
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
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
        if "check_metric_is_charts" not in ast.unparse(node)
        and not _forwards_metric(node)
        and not _is_refusal(node)
    ]
    assert not unguarded, (
        "these take a `metric` but neither check it against the chart nor "
        f"forward it to a verb that does: {unguarded}"
    )
