"""Regenerate the report's figures and results: python scripts/make_figures.py

Writes docs/figures/*.png and docs/results.json: the filtered and nominal
paths, the clearance over time, a sweep of the barrier rate ALPHA, and a
comparison of the exact 2D QP solver with SciPy's SLSQP.
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbf_safety_filter import control_barrier_function as m

FIGURES = ROOT / "docs" / "figures"
m.show_animation = False
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})
ALPHAS = [0.25, 0.5, 1.0, 1.5, 3.0, 6.0]


def run(alpha):
    m.ALPHA = alpha
    with contextlib.redirect_stdout(io.StringIO()):
        return m.main()


def nominal_path():
    p, path = m.START.copy(), [m.START.copy()]
    for _ in range(int(m.MAX_TIME / m.DT)):
        p = p + m.nominal_control(p) * m.DT
        path.append(p)
        if np.linalg.norm(m.GOAL - p) < m.GOAL_TOLERANCE:
            break
    return np.array(path)


def clearance(path):
    return np.linalg.norm(path[:, None] - m.OBSTACLES[:, :2], axis=-1) - m.KEEP_OUT


def draw_obstacles(ax):
    for (x, y, r), keep_out in zip(m.OBSTACLES, m.KEEP_OUT):
        ax.add_patch(Circle((x, y), r, color="0.45"))
        ax.add_patch(Circle((x, y), keep_out, fill=False, ls=":", color="0.3"))


def figure_paths(path, nominal):
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    draw_obstacles(ax)
    ax.plot(nominal[:, 0], nominal[:, 1], "--", color="0.5", label="nominal controller alone")
    ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=1.8, label="with the safety filter")
    ax.plot(*m.GOAL, "*", color="tab:green", ms=12, label="goal")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 10.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIGURES / "fig1_paths.png")
    plt.close(fig)


def figure_clearance(times, path, commands, nominal_commands):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for i, gap in enumerate(clearance(path).T, start=1):
        a1.plot(times, gap, label=f"obstacle {i}")
    a1.axhline(0.0, ls="--", color="tab:red", lw=1)
    a1.set_ylim(-0.3, 6.0)
    a1.set_xlabel("time [s]")
    a1.set_ylabel("clearance [m]")
    a1.legend(fontsize=7)
    change = np.linalg.norm(commands - nominal_commands, axis=1)
    a2.plot(times[1:], change, color="tab:purple")
    a2.set_xlabel("time [s]")
    a2.set_ylabel(r"$\|u - u_{\mathrm{nom}}\|$ [m/s]")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_clearance.png")
    plt.close(fig)


def figure_alpha(runs):
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    draw_obstacles(ax)
    for alpha, color in zip((0.25, 1.5, 6.0), ("tab:green", "tab:blue", "tab:orange")):
        path = runs[alpha][1]
        ax.plot(path[:, 0], path[:, 1], "-", color=color, lw=1.4, label=rf"$\alpha$ = {alpha}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 10.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIGURES / "fig3_alpha.png")
    plt.close(fig)


def solver_check(n=1000):
    rng = np.random.default_rng(1)
    worst, ours_t, slsqp_t = 0.0, 0.0, 0.0
    for _ in range(n):
        k = int(rng.integers(1, 5))
        a = rng.normal(size=(k, 2))
        b = -rng.uniform(0.0, 2.0, size=k)
        u_ref = rng.normal(scale=2.0, size=2)
        t0 = time.perf_counter()
        ours = m.solve_qp_2d(u_ref, a, b)
        t1 = time.perf_counter()
        ref = minimize(lambda u: np.sum((u - u_ref) ** 2), np.zeros(2),
                       jac=lambda u: 2.0 * (u - u_ref), method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda u: a @ u - b,
                                     "jac": lambda u: a}],
                       options={"ftol": 1e-14, "maxiter": 200}).x
        t2 = time.perf_counter()
        ours_t += t1 - t0
        slsqp_t += t2 - t1
        worst = max(worst, float(np.abs(ours - ref).max()))
    return {"problems": n, "max_difference": worst,
            "exact_solver_us": 1e6 * ours_t / n, "slsqp_us": 1e6 * slsqp_t / n}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = {alpha: run(alpha) for alpha in ALPHAS}
    m.ALPHA = 1.5
    times, path, commands, nominal_commands, h = runs[1.5]
    nominal = nominal_path()
    figure_paths(path, nominal)
    figure_clearance(times, path, commands, nominal_commands)
    figure_alpha(runs)
    sweep = {}
    for alpha, (t, p, u, un, hh) in runs.items():
        sweep[str(alpha)] = {"time_to_goal": float(t[-1]),
                             "min_clearance_m": float(clearance(p).min()),
                             "min_barrier": float(hh.min()),
                             "filtered_steps": int(np.any(np.abs(u - un) > 1e-12, axis=1).sum()),
                             "steps": len(u),
                             "mean_change": float(np.linalg.norm(u - un, axis=1).mean())}
    safe_nominal = [np.all(a @ un >= b) for (a, b), un in
                    zip((m.barrier_constraints(p) for p in path[:-1]), nominal_commands)]
    results = {"alpha_sweep": sweep,
               "nominal_min_barrier": float(min(m.barriers(p).min() for p in nominal)),
               "demo_safe_nominal_steps": int(np.sum(safe_nominal)),
               "demo_changed_safe_steps": int(np.sum(np.array(safe_nominal) & np.any(commands != nominal_commands, axis=1))),
               "solver_check": solver_check()}
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
