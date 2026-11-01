"""
CBF safety filter: the robot reaches its goal without ever entering an
obstacle, the filter only changes unsafe commands, and the small exact QP
solver agrees with a general-purpose optimiser.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from cbf_safety_filter import control_barrier_function as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def test_robot_reaches_the_goal(run):
    times, path, _, _, _ = run
    assert np.linalg.norm(path[-1] - m.GOAL) <= m.GOAL_TOLERANCE
    assert times[-1] < 25.0  # observed 19.2 s


def test_robot_never_enters_an_obstacle(run):
    _, path, _, _, h = run
    assert h.min() > 0.0  # observed 0.0095
    clearance = (np.linalg.norm(path[:, None] - m.OBSTACLES[:, :2], axis=-1)
                 - m.KEEP_OUT)
    assert clearance.min() > 0.0  # observed 4 mm


def test_without_the_filter_the_robot_would_collide():
    p, worst = m.START.copy(), np.inf
    for _ in range(int(m.MAX_TIME / m.DT)):
        p = p + m.nominal_control(p) * m.DT
        worst = min(worst, m.barriers(p).min())
    assert worst < -1.0  # observed -1.40: straight through obstacle 1


def test_filter_changes_only_unsafe_commands(run):
    _, path, commands, nominal, _ = run
    changed = np.any(commands != nominal, axis=1)
    safe = np.array([np.all(a @ u >= b) for (a, b), u in zip(
        (m.barrier_constraints(p) for p in path[:-1]), nominal)])
    assert safe.sum() > 50 and (~safe).sum() > 50  # observed 114 and 269
    assert not np.any(changed & safe)
    assert np.all(changed[~safe])


def test_constraint_rows_are_the_barrier_gradients():
    rng = np.random.default_rng(0)
    eps = 1e-6
    for p in rng.uniform(-2.0, 12.0, size=(10, 2)):
        a, _ = m.barrier_constraints(p)
        numeric = np.column_stack([
            (m.barriers(p + eps * e) - m.barriers(p - eps * e)) / (2 * eps)
            for e in np.eye(2)])
        np.testing.assert_allclose(a, numeric, atol=1e-6)


def test_qp_solver_matches_a_general_optimiser():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = rng.integers(1, 5)
        a = rng.normal(size=(n, 2))
        b = -rng.uniform(0.0, 2.0, size=n)  # u = 0 is feasible, as in use
        u_ref = rng.normal(scale=2.0, size=2)
        expected = minimize(
            lambda u: np.sum((u - u_ref) ** 2), np.zeros(2),
            jac=lambda u: 2.0 * (u - u_ref), method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda u: a @ u - b,
                          "jac": lambda u: a}],
            options={"ftol": 1e-14, "maxiter": 200}).x
        np.testing.assert_allclose(m.solve_qp_2d(u_ref, a, b), expected,
                                   atol=1e-5)

