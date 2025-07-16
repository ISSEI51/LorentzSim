import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable, Tuple


q = 1.0  # 荷電量
m = 1.0  # 質量
t0 = 0.0
t1 = 10.0  # 軌道表示の上限時刻
dt = 0.01
N = int(t1 / dt)


def numerical_derivative(
    f: Callable[[float], float], t: float, h: float = 1e-5
) -> float:
    return (f(t + h) - f(t - h)) / (2 * h)


def simulate_trajectory(r0, v0, B, E, dt, N):
    r = r0.copy()
    v = v0.copy()
    trajectory = [r.copy()]

    def acceleration(v):
        return (q / m) * (E + np.cross(v, B))

    for _ in range(N):
        k1_v = acceleration(v)
        k1_r = v
        k2_v = acceleration(v + 0.5 * dt * k1_v)
        k2_r = v + 0.5 * dt * k1_v
        k3_v = acceleration(v + 0.5 * dt * k2_v)
        k3_r = v + 0.5 * dt * k2_v
        k4_v = acceleration(v + dt * k3_v)
        k4_r = v + dt * k3_v
        v += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        r += (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        trajectory.append(r.copy())
    return np.array(trajectory)


# 追加: 最適化用の粗い点数
N_opt = 200  # 例: 200点で最適化
dt_opt = (t1 - t0) / N_opt


def fit_lorentz_trajectory_to_target(x_func, y_func, z_func):
    t_vals = np.linspace(t0, t1, N_opt + 1)
    target_traj = np.stack([x_func(t_vals), y_func(t_vals), z_func(t_vals)], axis=1)
    r0 = target_traj[0]

    def loss(params):
        v0 = params[0:3]
        B = params[3:6]
        E = params[6:9]
        sim_traj = simulate_trajectory(r0, v0, B, E, dt_opt, N_opt)
        return np.sum((sim_traj - target_traj) ** 2)

    v0_init = np.array(
        [
            (x_func(t0 + dt_opt) - x_func(t0)) / dt_opt,
            (y_func(t0 + dt_opt) - y_func(t0)) / dt_opt,
            (z_func(t0 + dt_opt) - z_func(t0)) / dt_opt,
        ]
    )
    x0 = np.concatenate([v0_init, np.zeros(3), np.zeros(3)])
    result = minimize(loss, x0, method="BFGS")
    v0_opt = result.x[0:3]
    B_opt = result.x[3:6]
    E_opt = result.x[6:9]
    return r0, v0_opt, B_opt, E_opt


def plot_trajectory_comparison(x_func, y_func, z_func, r0, v0, B, E, t0, t1, N):
    t_vals = np.linspace(t0, t1, N + 1)
    x_vals = np.array([x_func(t) for t in t_vals])
    y_vals = np.array([y_func(t) for t in t_vals])
    z_vals = np.array([z_func(t) for t in t_vals])
    sim_trajectory = simulate_trajectory(r0, v0, B, E, dt, N)
    x_sim, y_sim, z_sim = (
        sim_trajectory[:, 0],
        sim_trajectory[:, 1],
        sim_trajectory[:, 2],
    )
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_vals, y_vals, z_vals, label="Target trajectory", lw=2)
    ax.plot(x_sim, y_sim, z_sim, label="Simulated trajectory", lw=2, linestyle="--")
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], color="red", s=50, label="Start")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Target vs Simulated Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ----------------- 実行部 -----------------

if __name__ == "__main__":
    x = lambda t: t
    y = lambda t: 0 * t
    z = lambda t: 0 * t
    r0, v0, B, E = fit_lorentz_trajectory_to_target(x, y, z)
    print("初期位置 r0:", r0)
    print("初期速度 v0:", v0)
    print("磁場 B:", B)
    print("電場 E:", E)
    plot_trajectory_comparison(x, y, z, r0, v0, B, E, t0, t1, N)
