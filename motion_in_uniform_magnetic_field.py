import numpy as np
import matplotlib.pyplot as plt


"""
target of this program:
this program compare 3patterns of motion in uniform magnetic field.
the patterns is below
A. intial velocity: (v0cos, 0, v0sin), magnetic field: (0, 0, B)
B. intial velocity: (v0cos, 0, v0sin), magnetic field: (0, 0, B)
C. intial velocity: (v0cos, 0, v0sin), magnetic field: (0, 0, B)
"""
q = 1.0
m = 1.0
dt = 0.01
T = 20
N = int(T / dt)


# derivatives of the equations of motion based on Lorentz force
def derivatives(state: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return the time derivative of each variable.
    Args:
        state (np.array): 1D array of length 6 [x, y, z, vx, vy, vz]
                          osition (x, y, z) and velocity (vx, vy, vz), respectively
    Returns:
        np.array: 1D array of length 6 [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
                  Time derivative of each variable. The first half is the velocity vector and the second half is the acceleration vector.
    """
    x, y, z, vx, vy, vz = state
    v = np.array([vx, vy, vz])
    a = (q / m) * np.cross(v, B)
    return np.concatenate((v, a))


# numerical integration by Runge-Kutta quadratic method
def rk4(state0: np.ndarray, dt: float, N: int, B: np.ndarray) -> np.ndarray:
    """
    The RK4 method is used to evolve the state over time.
    Args:
        state0: initial state [x, y, z, vx, vy, vz]
        dt: time ticks
        N: number of steps
        B: magnetic field vector [Bx, By, Bz]
    Returns:
        sstate array (N+1, 6)
    """
    states = np.zeros((N + 1, 6))
    states[0] = state0

    for i in range(N):
        k1 = derivatives(states[i], B)
        k2 = derivatives(states[i] + 0.5 * dt * k1, B)
        k3 = derivatives(states[i] + 0.5 * dt * k2, B)
        k4 = derivatives(states[i] + dt * k3, B)
        states[i + 1] = states[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return states


def plot_trajectory(states: np.ndarray, title: str):
    """
    plot trajectry of x-y dimention
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label=title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# pattern A: Initial velocity has components in xy and z directions → herical motion
state_B = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
B_B = np.array([0.0, 0.0, 1.0])
states_B = rk4(state_B, dt, N, B_B)
plot_trajectory(states_B, "Pattern A: Helical Motion (B = [0, 0, 1])")

# pattern B: Magnetic field is in y direction → Motion is developed in x-z plane
state_D = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
B_D = np.array([0.0, 1.0, 0.0])
states_D = rk4(state_D, dt, N, B_D)
plot_trajectory(states_D, "Pattern B: Motion in xz-plane (B = [0, 1, 0])")

# pattern C: Magnetic field in any direction (x=y=z) → complex 3-D motion
state_E = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
B_E = np.array([1.0, 1.0, 1.0])
states_E = rk4(state_E, dt, N, B_E)
plot_trajectory(states_E, "Pattern C: 3D Helical Motion (B = [1, 1, 1])")
