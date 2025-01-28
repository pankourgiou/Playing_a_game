import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider, Button

# Constants for the "mood wavefunction"
hbar = 1  # Simplified Planck's constant
m = 1     # Simplified mass of the quantum cat

# Define a potential function: quantum cat's mood potential
def potential(x):
    return 0.5 * m * (x**2)  # Harmonic oscillator potential (for mood swings)

# Initialize parameters for the simulation
x_min, x_max = -5, 5  # Space boundaries
N = 500               # Number of points in the spatial grid
x = np.linspace(x_min, x_max, N)  # Spatial grid
dx = x[1] - x[0]      # Spatial step

# Initialize the wavefunction (a Gaussian)
sigma = 1.0
psi = np.exp(-x**2 / (2 * sigma**2))
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wavefunction

# Create the Hamiltonian matrix (discretized)
diagonal = hbar**2 / (2 * m * dx**2) + potential(x)
off_diagonal = -hbar**2 / (2 * m * dx**2) * np.ones(N-1)

H = np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)

# Time evolution parameters
dt = 0.01  # Time step
num_steps = 500  # Number of time steps

# Prepare for interactive plotting
plt.ion()
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
line, = ax.plot(x, np.abs(psi)**2, label='|Psi(x)|^2 (Mood Intensity)')
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.5)
ax.set_xlabel('Mood Space (x)')
ax.set_ylabel('Intensity')
ax.legend()
ax.set_title("Schrodinger's Cat Mood Simulation")

# Add sliders for interactivity
axcolor = 'lightgoldenrodyellow'
ax_sigma = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
sigma_slider = Slider(ax_sigma, 'Sigma', 0.1, 2.0, valinit=sigma)

# Add a reset button
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Function to update the plot based on the slider
def update(val):
    global psi
    sigma = sigma_slider.val
    psi = np.exp(-x**2 / (2 * sigma**2))
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize
    line.set_ydata(np.abs(psi)**2)
    fig.canvas.draw_idle()

sigma_slider.on_changed(update)

# Reset function
def reset(event):
    sigma_slider.reset()

button.on_clicked(reset)

# Time evolution loop (stops after user closes the plot)
try:
    for step in range(num_steps):
        # Compute the time evolution operator (simplified using Euler's method)
        psi = psi - 1j * dt / hbar * np.dot(H, psi)

        # Normalize the wavefunction
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

        # Update the plot
        line.set_ydata(np.abs(psi)**2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass

plt.ioff()
plt.show()

print("it's a playful cat!!")
