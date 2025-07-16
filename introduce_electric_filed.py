import numpy as np
import matplotlib.pyplot as plt

# constant
q = 1.0
m = 1.0
dt = 0.01
T = 20
N = int(T / dt)

# electric field
E = np.array([0.0, 0.0, 1.0])


# EOM by Lorentz force
def derivatives(state: np.ndarray, B: np.ndarray, E: np.ndarray) -> np.ndarray:
    x, y, z, vx, vy, vz = state
    v = np.array([vx, vy, vz])
    # add E term
    a = (q / m) * (E + np.cross(v, B))

    return np.concatenate((v, a))


def rk4(
    state0: np.ndarray, dt: float, N: int, B: np.ndarray, E: np.ndarray
) -> np.ndarray:
    states = np.zeros((N + 1, 6))
    states[0] = state0

    for i in range(N):
        k1 = derivatives(states[i], B, E)
        k2 = derivatives(states[i] + 0.5 * dt * k1, B, E)
        k3 = derivatives(states[i] + 0.5 * dt * k2, B, E)
        k4 = derivatives(states[i] + dt * k3, B, E)
        states[i + 1] = states[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return states


def plot_trajectory(states: np.ndarray, title: str):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(states[:, 0], states[:, 1], states[:, 2], label=title, color="blue")

    start = states[0]
    ax.scatter(start[0], start[1], start[2], color="red", s=50, label="Start")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()


# pattern A
state_A = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
B_A = np.array([0.0, 0.0, 1.0])
states_A = rk4(state_A, dt, N, B_A, E)
plot_trajectory(states_A, "PatternA + E(1,0,0)")

# pattern B
state_B = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
B_B = np.array([0.0, 0.0, 1.0])
states_B = rk4(state_B, dt, N, B_B, E)
plot_trajectory(states_B, "PatternB + E(1,1,1)")

# pattern C
state_C = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
B_C = np.array([1.0, 1.0, 1.0])
states_C = rk4(state_C, dt, N, B_C, E)
plot_trajectory(states_C, "PatternC + E(0,0,1)")
