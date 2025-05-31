import numpy as np
import matplotlib.pyplot as plt


"""
target of this program:
The objective is to simulate the motion of charged particles in a uniform magnetic field 
using different numerical solution methods (Eulerian and Runge-Kutta methods),
and to compare and evaluate their accuracy and stability
"""

# basic parameters(T: total simulation time, N: number of steps)
q = 1.0
m = 1.0
Bz = 1.0
dt = 0.01
T = 20
N = int(T / dt)

# initial state: circlar motion with radius1 and velocity1
state0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # x, y, z, vx, vy, vz


def derivatives(state):
    """_summary_
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
    B = np.array([0, 0, Bz])
    dvdt = (q / m) * np.cross(v, B)
    dxdt = v

    return np.concatenate((dxdt, dvdt))


# 1. Euler method
def euler(state0: np.array, dt: float, N: int):
    states = [state0]

    for _ in range(N):
        state = states[-1]
        new_state = state + dt * derivatives(state)
        states.append(new_state)

    return np.array(states)


# 2. Runge-Kutta 2 method
def rk2(state0: np.ndarray, dt: float, N: int) -> np.ndarray:
    states = [state0]
    for _ in range(N):
        state = states[-1]
        k1 = derivatives(state)
        k2 = derivatives(state + dt * k1 * 0.5)
        new_state = state + dt * (k1 + k2) * 0.5  # Modified update rule
        states.append(new_state)

    return np.array(states)


# 3. Runge-Kutta 4 method
def rk4(state0: np.array, dt: float, N: int):
    states = [state0]
    for _ in range(N):
        state = states[-1]
        k1 = derivatives(state)
        k2 = derivatives(state + dt * 0.5 * k1)
        k3 = derivatives(state + dt * 0.5 * k2)
        k4 = derivatives(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(new_state)

    return np.array(states)


# execute simulation
t = np.linspace(0, T, N + 1)
euler_states = euler(state0, dt, N)
rk2_states = rk2(state0, dt, N)
rk4_states = rk4(state0, dt, N)

# Add debugging prints
print("Shape of arrays:")
print("Euler states shape:", euler_states.shape)
print("RK2 states shape:", rk2_states.shape)
print("RK4 states shape:", rk4_states.shape)
print("\nFirst few points of RK2 trajectory:")
print("x:", rk2_states[0:5, 0])
print("y:", rk2_states[0:5, 1])


def kinetic_energy(states: np.array):
    # states is N+1*6 array, first parameter is the range of rows and second parameter is column.
    # v stores velocities of entire time.
    v = states[:, 3:6]

    return 0.5 * m * np.sum(v**2, axis=1)


# visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# xy orbit
axs[0].plot(
    euler_states[:, 0],
    euler_states[:, 1],
    label="Euler",
    color="red",
    linestyle="--",
    alpha=0.7,
)
axs[0].plot(
    rk2_states[:, 0],
    rk2_states[:, 1],
    label="RK2",
    color="green",
    linewidth=2,
    linestyle="-",
)
axs[0].plot(rk4_states[:, 0], rk4_states[:, 1], label="RK4", color="blue", alpha=0.7)
axs[0].set_title("Trajectory in xy-plane")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].axis("equal")
axs[0].grid(True)
axs[0].legend()

# kinetic energy
axs[1].plot(
    t,
    kinetic_energy(euler_states),
    label="Euler",
    color="red",
    linestyle="--",
    alpha=0.7,
)
axs[1].plot(
    t,
    kinetic_energy(rk2_states),
    label="RK2",
    color="green",
    linewidth=2,
    linestyle="-",
)
axs[1].plot(t, kinetic_energy(rk4_states), label="RK4", color="blue", alpha=0.7)
axs[1].set_title("Kinetic Energy Over Time")
axs[1].set_xlabel("Time[s]")
axs[1].set_ylabel("Energy[J]")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
