import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm

# Generate synthetic data with exponential decay background and a Gaussian signal
np.random.seed(42)

# Background: Exponential decay
x_background = np.linspace(60, 180, 500)
background_data = expon.pdf(x_background, scale=100)

# Signal: Gaussian peak
x_signal = np.linspace(60, 180, 500)  # Ensure the signal array has the same shape as the background
signal_data = norm.pdf(x_signal, loc=125, scale=10) * 0.004  # Adjust amplitude as needed

# Combine background and signal
y_data = background_data + signal_data

# Plot the line with exponential decay background and a small signal on top
with plt.xkcd():
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_background, y_data*10000, color='blue', label='Invariant Mass Distribution')
    ax.plot(x_signal, background_data*10000, '--', color='red', label='Background Fit')

    # Style the plot
    ax.set_xlim(60, 180)
    #ax.title('Invariant Mass Distribution with Exponential Decay and Signal')
    ax.set_xlabel('Invariant Mass')
    ax.set_ylabel('Events')
    ax.legend()
    ax.grid(True)

    # Show the plot
    fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/cartoon.png", bbox_inches='tight')
