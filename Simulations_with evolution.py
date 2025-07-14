import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.fft import fft, ifft

# Parameters
hbar = 1.0
m = 1.0
g = 0            # No Interaction strength
dt = 0.01           # Time step
t_max = 100         # Maximum time for simulation
L = 20.0            # x size
N = 1000            # Number of grid points
dx = L / N
x = np.linspace(-L/2, L/2 - dx, N)

# Initial wavefunction: ground state Gaussian
psi_ini = np.exp(-x**2)

# Normalize the initial wavefunction
psi_x = psi_ini / np.sqrt(np.sum(np.abs(psi_ini)**2) * dx)

# Store initial wavefunction for animation
data = [psi_x.copy()]

# Potential: harmonic trap
V_x = 0.5 * m * x**2

# k values
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Evolution operators
def kinetic(dt):
    return np.exp((1j * hbar * k**2 * dt) / (2 * m))

def nonlinear(psi_x, dt):
    return np.exp(-1j * (dt / hbar) * (V_x + g * np.abs(psi_x)**2))

# Split-Step Fourier Method evolution
def evolve(psi_x, dt):
    psi_1 = fft(psi_x)
    psi_2 = psi_1 * kinetic(dt / 2)
    psi_3 = ifft(psi_2)
    psi_4 = psi_3 * nonlinear(psi_3, dt)
    psi_5 = fft(psi_4)
    psi_6 = psi_5 * kinetic(dt / 2)
    psi_x = ifft(psi_6)
    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
    return psi_x / norm

# Simulation loop
timesteps = int(t_max / dt)
for i in range(timesteps):
    psi_x = evolve(psi_x, dt)
    data.append(psi_x.copy())

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(data[0])**2, label=r"$|\psi(x,t)|^2$")
pot_line, = ax.plot(x, V_x, 'r--', label="Harmonic Potential (scaled)")
ax.set_ylim(0, 1)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
title = ax.set_title("")
ax.legend(loc="upper right")

def animate(i):
    line.set_ydata(np.abs(data[i])**2)
    title.set_text(f"t = {i * dt:.2f}")
    return line, pot_line, title

ani = animation.FuncAnimation(fig, animate, frames=len(data), interval=10)


plt.show()
