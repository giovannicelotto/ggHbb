import ROOT
import numpy as np
from scipy.integrate import quad
def DoubleSidedCrystalballFunction(x, par):
    alpha_l = par[0]
    alpha_r = par[1]
    n_l = par[2]
    n_r = par[3]
    mean = par[4]
    sigma = par[5]
    N = par[6]
    

    t = (x[0] - mean) / sigma
    result = 0.0
    
    fact1TLessMinosAlphaL = alpha_l / n_l
    fact2TLessMinosAlphaL = (n_l / alpha_l) - alpha_l - t
    
    fact1THihgerAlphaH = alpha_r / n_r
    fact2THigherAlphaH = (n_r / alpha_r) - alpha_r + t
    
    if -alpha_l <= t <= alpha_r:
        result = np.exp(-0.5 * t * t) 
    elif t < -alpha_l:
        result = np.exp(-0.5 * alpha_l * alpha_l) * np.power(fact1TLessMinosAlphaL * fact2TLessMinosAlphaL, -n_l) 
    elif t > alpha_r:
        result = np.exp(-0.5 * alpha_r * alpha_r) * np.power(fact1THihgerAlphaH * fact2THigherAlphaH, -n_r)
    
    return N * result

def DoubleSidedCrystalballFunctionPDF(x, alpha_l, alpha_r, n_l, n_r, mean, sigma):
    t = (x[0] - mean) / sigma
    
    if -alpha_l <= t <= alpha_r:
        result = np.exp(-0.5 * t * t)
    elif t < -alpha_l:
        A_l = (n_l / np.abs(alpha_l)) ** n_l * np.exp(-0.5 * alpha_l * alpha_l)
        B_l = n_l / np.abs(alpha_l) - np.abs(alpha_l) - t
        result = A_l * np.power(B_l, -n_l)
    elif t > alpha_r:
        A_r = (n_r / np.abs(alpha_r)) ** n_r * np.exp(-0.5 * alpha_r * alpha_r)
        B_r = n_r / np.abs(alpha_r) - np.abs(alpha_r) + t
        result = A_r * np.power(B_r, -n_r)
    else:
        result = 0.0
    
    return result

def integrand(t, alpha_l, alpha_r, n_l, n_r, sigma):
    if -alpha_l <= t <= alpha_r:
        return np.exp(-0.5 * t * t)
    elif t < -alpha_l:
        A_l = (n_l / np.abs(alpha_l)) ** n_l * np.exp(-0.5 * alpha_l * alpha_l)
        B_l = n_l / np.abs(alpha_l) - np.abs(alpha_l) - t
        return A_l * np.power(B_l, -n_l)
    elif t > alpha_r:
        A_r = (n_r / np.abs(alpha_r)) ** n_r * np.exp(-0.5 * alpha_r * alpha_r)
        B_r = n_r / np.abs(alpha_r) - np.abs(alpha_r) + t
        return A_r * np.power(B_r, -n_r)
    else:
        return 0.0

def normalize_constant(alpha_l, alpha_r, n_l, n_r, sigma):
    integral, _ = quad(lambda t: integrand(t, alpha_l, alpha_r, n_l, n_r, sigma), -np.inf, np.inf)
    return 1.0 / integral

def DoubleSidedCrystalballFunctionPDF_normalized(x, alpha_l, alpha_r, n_l, n_r, mean, sigma):
    norm_const = normalize_constant(alpha_l, alpha_r, n_l, n_r, sigma)
    return norm_const * DoubleSidedCrystalballFunctionPDF(x, alpha_l, alpha_r, n_l, n_r, mean, sigma)

