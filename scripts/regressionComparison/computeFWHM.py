import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def compute_fwhm(x, y):
    """
    Compute the Full Width at Half Maximum (FWHM) of a histogram or curve.
    
    Parameters:
    x: array-like
        The x values of the histogram (e.g., bin centers).
    y: array-like
        The y values of the histogram (e.g., bin counts).
        
    Returns:
    fwhm: float
        The Full Width at Half Maximum (FWHM).
    """
    # Find the maximum value and half of that value
    max_y = np.max(y)
    half_max_y = max_y / 2.0
    
    # Find the indices of the maximum point
    max_idx = np.argmax(y)
    
    # Interpolation to find the positions of the half max values
    f = interp1d(x, y - half_max_y, kind='linear')
    
    # Find the two points where the curve crosses the half maximum
    # This will occur on either side of the peak (max value)
    
    # Going left from the peak (find where the curve crosses half max)
    left_half_max_idx = np.where(y[:max_idx] < half_max_y)[0][-1]
    
    # Going right from the peak (find where the curve crosses half max)
    right_half_max_idx = max_idx + np.where(y[max_idx:] < half_max_y)[0][0]
    
    # x-values of the two points where it crosses the half-maximum
    x1 = x[left_half_max_idx]
    x2 = x[right_half_max_idx]
    
    # Compute the FWHM
    fwhm = x2 - x1
    
    return fwhm, x1, x2

# Example usage
x = np.linspace(0, 10, 100)  # bin centers
y = np.exp(-((x - 5) ** 2) / (2 * 0.5 ** 2))  # Gaussian curve for demonstration

# Compute FWHM
fwhm, x1, x2 = compute_fwhm(x, y)

print(f"FWHM: {fwhm}")
