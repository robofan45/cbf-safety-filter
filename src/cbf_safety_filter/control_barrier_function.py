"""

Control barrier function (CBF) safety filter

author: Mahim Trivedi

A robot heads straight for its goal (the nominal controller), and a safety
filter changes that command as little as possible so the robot never enters
an obstacle. Each obstacle i gives a barrier h_i(p) = |p - c_i|^2 - R_i^2,
positive while the robot is clear of it. Every step the filter solves the
quadratic program

    minimise |u - u_nominal|^2  subject to  dh_i/dt >= -ALPHA * h_i  for all i

so a barrier may shrink, but more and more slowly as it nears zero, and it
never crosses zero. A command that is already safe passes unchanged.

The robot moves as a point with velocity u (its size is added to each
obstacle radius). h is convex in p, so even the discrete step p + u * DT
keeps h_i >= (1 - ALPHA * DT) * h_i > 0: the guarantee is exact.

Reference:
- Ames et al., "Control Barrier Functions: Theory and Applications",
  European Control Conference, 2019.
- Ames, Xu, Grizzle and Tabuada, "Control Barrier Function Based Quadratic
  Programs for Safety Critical Systems", IEEE Transactions on Automatic
  Control, 2017.

"""
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

DT = 0.05  # [s]
ALPHA = 1.5  # [1/s] how fast a barrier may shrink, relative to its value
K_GOAL = 1.0  # [1/s] gain of the nominal go-to-goal controller
MAX_SPEED = 1.0  # [m/s] of the nominal controller
ROBOT_RADIUS = 0.3  # [m]
GOAL_TOLERANCE = 0.05  # [m]
MAX_TIME = 30.0  # [s]

START = np.array([0.0, 0.0])
GOAL = np.array([10.0, 0.8])
# obstacle centre x, y and radius [m]
OBSTACLES = np.array([[3.0, -0.3, 1.0],
                      [6.0, 1.0, 0.8],
                      [8.5, 0.3, 0.6]])
KEEP_OUT = OBSTACLES[:, 2] + ROBOT_RADIUS  # [m] closest the robot centre may come

show_animation = True


def nominal_control(p: np.ndarray) -> np.ndarray:
    """Velocity straight at the goal, slowing down near it, at most
    MAX_SPEED. It knows nothing about obstacles."""
    u = K_GOAL * (GOAL - p)
    speed = math.hypot(u[0], u[1])
    return u if speed <= MAX_SPEED else u * (MAX_SPEED / speed)


def barriers(p: np.ndarray) -> np.ndarray:
    """h_i = |p - c_i|^2 - KEEP_OUT_i^2 for each obstacle (M,)."""
    return np.sum((p - OBSTACLES[:, :2]) ** 2, axis=1) - KEEP_OUT ** 2


def barrier_constraints(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows a_i and bounds b_i of the safety constraints a_i . u >= b_i:
    dh_i/dt = 2 (p - c_i) . u must be at least -ALPHA * h_i."""
    return 2.0 * (p - OBSTACLES[:, :2]), -ALPHA * barriers(p)


def solve_qp_2d(u_ref: np.ndarray, a: np.ndarray,
                b: np.ndarray) -> np.ndarray:
    """Minimise |u - u_ref|^2 subject to a @ u >= b, for a 2D u.

    The minimum of this convex problem is the unconstrained optimum, the
    projection onto one constraint line, or a corner where two constraint
    lines meet, so checking those few candidates gives the exact answer."""
    candidates = [u_ref]
    for a_i, b_i in zip(a, b):
        candidates.append(u_ref + (b_i - a_i @ u_ref) / (a_i @ a_i) * a_i)
    for i, j in itertools.combinations(range(len(b)), 2):
        pair = a[[i, j]]
        if abs(np.linalg.det(pair)) > 1e-12:
            candidates.append(np.linalg.solve(pair, b[[i, j]]))
    feasible = [u for u in candidates if np.all(a @ u >= b - 1e-9)]
    return min(feasible, key=lambda u: float(np.sum((u - u_ref) ** 2)))


def safety_filter(p: np.ndarray, u_nominal: np.ndarray) -> np.ndarray:
    """The safe command closest to the nominal one. While the robot is
    clear of every obstacle, u = 0 meets all constraints, so the problem is
    always feasible."""
    a, b = barrier_constraints(p)
    return solve_qp_2d(u_nominal, a, b)


def plot_frame(ax_map: plt.Axes, ax_gap: plt.Axes, path: np.ndarray,
               u: np.ndarray, u_nominal: np.ndarray, times: np.ndarray,
               clearance: np.ndarray) -> None:  # pragma: no cover
    p = path[-1]
    ax_map.cla()
    ax_map.plot(*np.column_stack([START, GOAL]), "--", color="0.6",
                label="nominal path")
    for (x, y, radius), keep_out in zip(OBSTACLES, KEEP_OUT):
        ax_map.add_patch(Circle((x, y), radius, color="0.35"))
        ax_map.add_patch(Circle((x, y), keep_out, fill=False, ls=":",
                                color="0.35"))
    ax_map.plot(path[:, 0], path[:, 1], "-b", lw=2, label="filtered path")
    ax_map.add_patch(Circle((p[0], p[1]), ROBOT_RADIUS, color="tab:blue"))
    ax_map.arrow(p[0], p[1], 1.5 * u_nominal[0], 1.5 * u_nominal[1],
                 head_width=0.15, color="0.5", label="nominal command")
    ax_map.arrow(p[0], p[1], 1.5 * u[0], 1.5 * u[1], head_width=0.15,
                 color="tab:blue", label="safe command")
    ax_map.plot(*GOAL, "*", color="tab:green", ms=14, label="goal")
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-1.0, 11.0)
    ax_map.set_ylim(-2.5, 3.0)
    ax_map.grid(True)
    ax_map.legend(loc="lower right", fontsize=8)
    ax_map.set_title("CBF safety filter")

    ax_gap.cla()
    for i, gap in enumerate(clearance.T, start=1):
        ax_gap.plot(times, gap, label=f"obstacle {i}")
    ax_gap.axhline(0.0, color="tab:red", ls="--", label="collision")
    ax_gap.set_xlim(0.0, max(times[-1], 1.0))
    ax_gap.set_ylim(-0.5, 6.0)
    ax_gap.set_xlabel("time [s]")
    ax_gap.set_ylabel("clearance [m]")
    ax_gap.grid(True)
    ax_gap.legend(loc="upper right", fontsize=8)
    ax_gap.set_title("distance to each keep-out circle")
    plt.pause(0.001)


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                    np.ndarray]:
    print(__file__ + " start!!")
    p = START.copy()
    times, path, h = [0.0], [p], [barriers(p)]
    commands, nominal_commands = [], []
    if show_animation:  # pragma: no cover
        fig, (ax_map, ax_gap) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                           else None])

    while (math.hypot(*(GOAL - p)) > GOAL_TOLERANCE
           and times[-1] < MAX_TIME):
        u_nominal = nominal_control(p)
        u = safety_filter(p, u_nominal)
        p = p + u * DT
        times.append(times[-1] + DT)
        path.append(p)
        h.append(barriers(p))
        commands.append(u)
        nominal_commands.append(u_nominal)

        if show_animation:  # pragma: no cover
            clearance = (np.linalg.norm(np.array(path)[:, None]
                                        - OBSTACLES[:, :2], axis=-1)
                         - KEEP_OUT)
            plot_frame(ax_map, ax_gap, np.array(path), u, u_nominal,
                       np.array(times), clearance)

    if show_animation:  # pragma: no cover
        plt.show()

    return (np.array(times), np.array(path), np.array(commands),
            np.array(nominal_commands), np.array(h))


if __name__ == '__main__':
    main()
