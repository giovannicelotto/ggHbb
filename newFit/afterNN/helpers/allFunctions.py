import yaml
import numpy as np
from scipy.special import erf

#with open("/t3home/gcelotto/newFit/afterNN/config_1.yaml", "r") as f:
#    params = yaml.safe_load(f)  # Read as dictionary
#
#x1 = params["x1"]
#x2 = params["x2"]
#key = params["key"]
#nbins = params["nbins"]

def set_x_bounds(x1_val, x2_val):
    """Function to set global x1 and x2 values."""
    global x1, x2
    x1, x2 = x1_val, x2_val


def pol1N(x, b):
    return 1 + b*x

def continuum_background_1(x, norm, B, b):
    maxX, minX = x2, x1
    
    def indef_integral(x, B, b):
        term1 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term2 = -np.exp(-B*x) / B
        integral = term1 + term2
        return integral
    
    integral = indef_integral(maxX, B, b) - indef_integral(minX, B, b)
    return norm * pol1N(x, b) * np.exp(-B*x) / integral


# Pol2 * expo Normalized
def pol2N(x,  b, c):
    # norm inside the continuum bkg function
    return 1 + b*x + c*x**2
def expo(x,  B):
    return  np.exp(-B * x)
def continuum_background_2(x, norm, B,  b, c):
    maxX, minX = x2, x1
    def indef_integral(x, B, b, c):
        term1 = c* (-x**2 * np.exp(-B*x) / B - 2*x * np.exp(-B*x) / B**2 - 2 * np.exp(-B*x) / B**3)
        term2 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term3 = -  np.exp(-B*x) / B
        integral =  (term1 + term2 + term3)
        return integral
    integral = indef_integral(maxX, B, b, c) - indef_integral(minX, B, b, c)
    return norm * pol2N(x,  b, c)*expo(x,  B)/integral


def pol3N(x, b, c, d):
    # Normalized third-degree polynomial
    return 1 + b*x + c*x**2 + d*x**3

def continuum_background_3(x, norm, B, b, c, d):
    maxX, minX = x2, x1
    
    def indef_integral(x, B, b, c, d):
        term1 = d * (6 / B**4 + 6*x / B**3 + 3*x**2 / B**2 + x**3 / B) * np.exp(-B*x)
        term2 = c * (-x**2 * np.exp(-B*x) / B - 2*x * np.exp(-B*x) / B**2 - 2 * np.exp(-B*x) / B**3)
        term3 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term4 = -np.exp(-B*x) / B
        integral = term1 + term2 + term3 + term4
        return integral
    
    integral = indef_integral(maxX, B, b, c, d) - indef_integral(minX, B, b, c, d)
    return norm * pol3N(x, b, c, d) * expo(x, B) / integral

def pol1N_Z(x, p1):
    result = 1 + p1 * x
    integral = (x2-x1) + p1*(x2**2-x1**2)/2
    return  result/integral


def dscb(x, mean, sigma, alphaL, nL, alphaR, nR):
    t = (x - mean) / sigma
    A = (nL / abs(alphaL)) ** nL * np.exp(-0.5 * alphaL ** 2)
    B = (nR / abs(alphaR)) ** nR * np.exp(-0.5 * alphaR ** 2)

    left = A * (nL / abs(alphaL) - abs(alphaL) - t) ** (-nL)  # Left tail
    expr = nR / abs(alphaR) - abs(alphaR) + t
    expr =np.where(expr>0,expr, 1e-12)
    right = B * expr ** (-nR)

    central = np.exp(-0.5 * t ** 2)


    integralLeft =  -A*sigma/(-nL+1)*(
        (nL / abs(alphaL) - abs(alphaL) - (-alphaL)) ** (-nL+1) - 
        (nL / abs(alphaL) - abs(alphaL) - (x1-mean)/sigma) ** (-nL+1))
    integralRight =  B*sigma/(-nR+1)*(
        (nR / abs(alphaR) - abs(alphaR) + (x2-mean)/sigma) ** (-nR+1) - 
        (nR / abs(alphaR) - abs(alphaR) + (alphaR)) ** (-nR+1))
    integralCentral = 1/2*(erf(alphaR/np.sqrt(2))  -    erf(-alphaL/np.sqrt(2)))*np.sqrt(np.pi)*(np.sqrt(2)*sigma)
    totalIntegral = integralLeft + integralRight + integralCentral
    return np.where(t < -alphaL, left, np.where(t > alphaR, right, central))/totalIntegral

def gaussianN(x, mean, sigmaG):
    integralGauss = 1/2*(erf((x2-mean)/(sigmaG*np.sqrt(2)))  -    erf((x1-mean)/(sigmaG*np.sqrt(2))))
    return  1/(np.sqrt(2*np.pi)*sigmaG)* np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)/integralGauss

def zPeak(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return normSig*(fraction_dscb*dscb(x, mean, sigma, alphaL, nL, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))
def continuum_plus_Z_1(x, normBkg, B, b, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return continuum_background_1(x, normBkg, B,  b) + zPeak(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def continuum_plus_Z_2(x, normBkg, B, b, c, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return continuum_background_2(x, normBkg, B,  b, c) + zPeak(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def continuum_plus_Z_3(x, normBkg, B, b, c, d, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return continuum_background_3(x, normBkg, B,  b, c, d) + zPeak(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)



