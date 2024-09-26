from scipy.special import erf
import numpy as np
# Double-sided Crystal Ball function
def double_sided_crystal_ball(x, mean, sigma, alphaL, nL, alphaR, nR):
    """ Double-sided Crystal Ball function """
    A = np.exp(-0.5 * alphaL**2)
    B = nL / abs(alphaL) - abs(alphaL)
    C = np.exp(-0.5 * alphaR**2)
    D = nR / abs(alphaR) - abs(alphaR)
    
    result = np.zeros_like(x)
    # Left tail
    maskL = (x - mean) / sigma < -alphaL
    result[maskL] = A * (B - (x[maskL] - mean) / sigma) ** -nL
    
    # Gaussian core
    maskCore = (x - mean) / sigma >= -alphaL
    maskCore &= (x - mean) / sigma <= alphaR
    result[maskCore] = np.exp(-0.5 * ((x[maskCore] - mean) / sigma) ** 2)
    
    # Right tail
    maskR = (x - mean) / sigma > alphaR
    result[maskR] = C * (D + (x[maskR] - mean) / sigma) ** -nR

    return result
