import numpy as np
from scipy.special import erf
import scipy.integrate as integrate


# Essential Functions

def set_x_bounds(x1_val, x2_val):
    """Function to set global x1 and x2 values."""
    global x1, x2
    x1, x2 = x1_val, x2_val

# Polynomials without norm parameters
def pol1N(x, p1):
    return 1 + p1*x

def pol2N(x,  b, c):
    return 1 + b*x + c*x**2

def pol3N(x, b, c, d):
    return 1 + b*x + c*x**2 + d*x**3



##
##
##          Z Functions
##
## 


# Used functions for Z
def dscb(x, mean, sigma, alphaL, nL, alphaR, nR):
    t = (x - mean) / sigma
    A = ((nL / abs(alphaL)) ** nL) * np.exp(-0.5 * alphaL ** 2)
    B = ((nR / abs(alphaR)) ** nR) * np.exp(-0.5 * alphaR ** 2)

    #Avoid negative base
    base_left = (nL / abs(alphaL) - abs(alphaL) - t)
    base_left = np.where(base_left > 1e-12, base_left, 1e-12)
    left = A * base_left ** (-nL)

    base_right = nR / abs(alphaR) - abs(alphaR) + t
    base_right =np.where(base_right>0,base_right, 1e-12)
    #base_right = np.where(base_right < 100, base_right, 100)
    right = B * base_right ** (-nR)

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


def rscb(x, mean, sigma, alphaR, nR):
    t = (x - mean) / sigma
    B = (nR / abs(alphaR)) ** nR * np.exp(-0.5 * alphaR ** 2)

    expr = nR / abs(alphaR) - abs(alphaR) + t
    expr =np.where(expr>0,expr, 1e-12)
    
    right = B * expr ** (-nR)
    central = np.exp(-0.5 * t ** 2)

    integralRight =  B*sigma/(-nR+1)*(
        (nR / abs(alphaR) - abs(alphaR) + (x2-mean)/sigma) ** (-nR+1) - 
        (nR / abs(alphaR) - abs(alphaR) + (alphaR)) ** (-nR+1))
    integralCentral = 1/2*(erf(alphaR/np.sqrt(2))  -    erf((x1-mean)/sigma/np.sqrt(2)))*np.sqrt(np.pi)*(np.sqrt(2)*sigma)
    totalIntegral = integralRight + integralCentral
    return np.where(t < alphaR,central, right)/totalIntegral

def gaussianN(x, mean, sigmaG):
    integralGauss = 1/2*(erf((x2-mean)/(sigmaG*np.sqrt(2)))  -    erf((x1-mean)/(sigmaG*np.sqrt(2))))
    return  1/(np.sqrt(2*np.pi)*sigmaG)* np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)/integralGauss

def zPeak_rscb(x, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))

def zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return normSig*(fraction_dscb*dscb(x, mean, sigma, alphaL, nL, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))


def zPeak_rscb_pol1(x, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG, fractionG, p1):
    return normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (fractionG)*gaussianN(x, mean, sigmaG) + (1-fractionG-fraction_dscb)*pol1N(x, p1))






##
##
##    Bkg only functions  
##
##







def expo(x,  B):
    return  np.exp(-B * x)

def exp_pol1(x, norm, B, b):
    maxX, minX = x2, x1
    
    def indef_integral(x, B, b):
        term1 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term2 = -np.exp(-B*x) / B
        integral = term1 + term2
        return integral
    
    integral = indef_integral(maxX, B, b) - indef_integral(minX, B, b)
    return norm * pol1N(x, b) * np.exp(-B*x) / integral

def exp_pol2(x, norm, B, b, c):
    ''' normalized pol2 * exp'''
    def unnormalized_function(x):
        pol2 = c * x**2 + b * x + 1 
        exp = np.exp(-B * x)
        return (exp) * pol2
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    return normalization_factor * unnormalized_function(x)

#def exp_pol2(x, norm, B,  b, c):
#    maxX, minX = x2, x1
#    def indef_integral(x, B, b, c):
#        term1 = c* (-x**2 * np.exp(-B*x) / B - 2*x * np.exp(-B*x) / B**2 - 2 * np.exp(-B*x) / B**3)
#        term2 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
#        term3 = -  np.exp(-B*x) / B
#        integral =  (term1 + term2 + term3)
#        return integral
#    integral = indef_integral(maxX, B, b, c) - indef_integral(minX, B, b, c)
#    return norm * pol2N(x,  b, c)*expo(x,  B)/integral


def expExp_pol2(x, norm, B, C, b, c):
    def unnormalized_function(x):
        pol2 = c * x**2 + b * x + 1  # Quadratic polynomial
        exp1 = np.exp(-B * x)
        exp2 = np.exp(-C * x)
        return (exp1 + exp2) * pol2
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)

def expPol2_expPol2(x, norm, B, C, p1, p2, p11, p12, f):
    def unnormalized_function(x):
        pol2 = p2 * x**2 + p1 * x + 1  
        pol2_ = p12 * x**2 + p11 * x + 1 
        exp1 = np.exp(-B * x)
        exp2 = np.exp(-C * x)
        return (f)*(exp1 * pol2) + (1-f)*(exp2*pol2_)
    
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)

def expExp_pol2_turnOn(x, norm, B, C, b, c, aa, bb, f):
    def unnormalized_function(x):
        pol2 = c * x**2 + b * x + 1  # Quadratic polynomial
        exp1 = np.exp(-B * x)
        exp2 = np.exp(-C * x)
        turnOn = bb * x**2 + aa * x + 1  # Quadratic polynomial
        return f*(exp1 + exp2) * pol2 + (1-f)*turnOn
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)

def expExp_pol2_turnOnPol1(x, norm, B, C, b, c, aa, f):
    def unnormalized_function(x):
        pol2 = c * x**2 + b * x + 1  # Quadratic polynomial
        exp1 = np.exp(-B * x)
        exp2 = np.exp(-C * x)
        turnOn =  aa * x + 1  # Quadratic polynomial
        return f*(exp1 + exp2) * pol2 + (1-f)*turnOn
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)

def expExp_pol2_turnOnPol3(x, norm, B, C, b, c,d, aa, bb, f):
    def unnormalized_function(x):
        pol3 = d*x**3 + c * x**2 + b * x + 1  # Quadratic polynomial
        exp1 = np.exp(-B * x)
        exp2 = np.exp(-C * x)
        turnOn = bb * x**2 + aa * x + 1  # Quadratic polynomial
        return f*(exp1 + exp2) * pol3 + (1-f)*turnOn
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)


def exp_pol2_turnOn3(x, norm, B, b, c, aa, bb, cc, f):
    def unnormalized_function(x):
        pol2 = c * x**2 + b * x + 1  # Quadratic polynomial
        exp1 = np.exp(-B * x)
        turnOn = cc*x**3+bb * x**2 + aa * x + 1  # Quadratic polynomial
        return f*(exp1) * pol2 + (1-f)*turnOn
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)

def exp_gaus_turnOn(x, norm, B, b, c, mu_to, sigma_to, f):
    def unnormalized_function(x):
        pol2 = c * x**2 + b * x + 1  # Quadratic polynomial
        exp1 = np.exp(-B * x)
        turnOn = np.exp(-0.5*(x-mu_to)**2/sigma_to**2)
        return f*(exp1) * pol2 + (1-f)*turnOn
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)




def kinThreshold(x, norm, B, m0, p1, p2):
    def unnormalized_function(x):
        pol2 = p2 * (x-m0)**2 + p1 * (x-m0) + 1
        exp1 = np.exp(-B * (x-m0))
        return (exp1) * pol2 
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / integral_value
    
    return normalization_factor * unnormalized_function(x)


def exp_pol3(x, norm, B, b, c, d):
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

def exp_pol5(x, norm, B, b, c, d, e, f):
    maxX, minX = x2, x1
    
    def indef_integral(x, B, b, c, d, e, f):
        term1 = f * (120 / B**5 + 120*x / B**4 + 60*x**2 / B**3 + 20*x**3 / B**2 + 5*x**4 / B + x**5) * np.exp(-B*x)
        term2 = e * (-x**4 * np.exp(-B*x) / B - 4*x**3 * np.exp(-B*x) / B**2 - 12*x**2 * np.exp(-B*x) / B**3 - 24*x * np.exp(-B*x) / B**4 - 24 * np.exp(-B*x) / B**5)
        term3 = d * (-x**3 * np.exp(-B*x) / B - 3*x**2 * np.exp(-B*x) / B**2 - 6*x * np.exp(-B*x) / B**3 - 6 * np.exp(-B*x) / B**4)
        term4 = c * (-x**2 * np.exp(-B*x) / B - 2*x * np.exp(-B*x) / B**2 - 2 * np.exp(-B*x) / B**3)
        term5 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term6 = -np.exp(-B*x) / B
        integral = term1 + term2 + term3 + term4 + term5 + term6
        return integral
    
    integral = indef_integral(maxX, B, b, c, d, e, f) - indef_integral(minX, B, b, c, d, e, f)
    
    def pol5N(x, b, c, d, e, f):
        return b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + 1  # Including the constant term for normalization
    
    
    return norm * pol5N(x, b, c, d, e, f) * expo(x, B) / integral







# Bkg only
def expExpExp(x, normBkg, B, C, D):
    return normBkg*expo(x, B)*expo(x, C)*expo(x, D)





# Functions for Data
# DSCB
def continuum_DSCB1(x, normBkg, B, b, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return exp_pol1(x, normBkg, B,  b) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def continuum_DSCB2(x, normBkg, B, b, c, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return exp_pol2(x, normBkg, B,  b, c) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def continuum_DSCB3(x, normBkg, B, b, c, d, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return exp_pol3(x, normBkg, B,  b, c, d) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def continuum_DSCB5(x, normBkg, B, b, c, d, e, f, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return exp_pol5(x, normBkg, B,  b, c, d, e, f) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def expExpPol2_DSCB(x, normBkg, B, C, b, c, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return expExp_pol2(x, normBkg, B, C, b, c) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)



# Exp Exp
def expExpExp_DSCB(x, normBkg, B, C, D, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG):
    return normBkg*(expo(x, B)*expo(x, C)*expo(x, D)) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

# RSCB
def continuum_RSCB1(x, normBkg, B, b, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return exp_pol1(x, normBkg, B,  b) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))
def continuum_RSCB2(x, normBkg, B, b, c, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return exp_pol2(x, normBkg, B,  b, c) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))
def continuum_RSCB3(x, normBkg, B, b, c, d, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return exp_pol3(x, normBkg, B,  b, c, d) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))

def expExpPol2_RSCB(x, normBkg, B, C, b, c, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return expExp_pol2(x, normBkg, B, C, b, c) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))

def expExp_pol2_turnOn_RSCB(x, normBkg, B, C, b, c, aa, bb, f, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return expExp_pol2_turnOn(x, normBkg, B, C, b, c, aa, bb, f) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))
def exp_pol2_turnOn3_RSCB(x, normBkg, B,  b, c, aa, bb, cc, f, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return exp_pol2_turnOn3(x, normBkg, B,  b, c, aa, bb,cc, f) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))
def exp_gaus_turnOn_RSCB(x, normBkg, B, b, c, mu_to, sigma_to, f, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return exp_gaus_turnOn(x, normBkg, B, b, c,mu_to=mu_to, sigma_to=sigma_to, f=f) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))
def kinThreshold_RSCB(x, normBkg, B, m0, p1, p2, normSig, fraction_dscb, mean, sigma, alphaR, nR, sigmaG):
    return kinThreshold(x, normBkg, B, m0, p1, p2) + normSig*(fraction_dscb*rscb(x, mean, sigma, alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))


def expExp_pol2_turnOn_DSCB(x, normBkg, B, C, b, c, aa, bb, f, normSig, fraction_dscb, mean, sigma,  alphaL, nL, alphaR, nR, sigmaG):
    return expExp_pol2_turnOn(x, normBkg, B, C, b, c, aa, bb, f) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)
def expExp_pol2_turnOnPol3_DSCB(x, normBkg, B, C, b, c, d, aa, bb, f, normSig, fraction_dscb, mean, sigma,  alphaL, nL, alphaR, nR, sigmaG):
    return expExp_pol2_turnOnPol3(x, normBkg, B, C, b, c, d, aa, bb, f) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)
def exp_pol2_turnOn3_DSCB(x, normBkg, B, b, c, aa, bb, cc, f, normSig, fraction_dscb, mean, sigma,  alphaL, nL, alphaR, nR, sigmaG):
    return exp_pol2_turnOn3(x, normBkg, B,  b, c, aa, bb,cc, f) + zPeak_dscb(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, sigmaG)

def exp_gaus_turnOn_DSCB(x, normBkg, B, b, c, mu_to, sigma_to, f, normSig, fraction_dscb, mean, sigma,  alphaL, nL,alphaR, nR, sigmaG):
    return exp_gaus_turnOn(x, normBkg, B, b, c,mu_to, sigma_to, f) + normSig*(fraction_dscb*dscb(x, mean, sigma,  alphaL, nL,alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))


def expPol2_expPol2_DSCB(x, normBkg, B, C, p1,p2, p11, p12, f, normSig, fraction_dscb, mean, sigma,  alphaL, nL,alphaR, nR, sigmaG):
        return expPol2_expPol2(x, normBkg, B, C,p1,p2, p11, p12, f) + normSig*(fraction_dscb*dscb(x, mean, sigma,  alphaL, nL,alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))

def expExp_pol2_turnOnPol1_DSCB(x, normBkg, B, C, b,c, aa, f, normSig, fraction_dscb, mean, sigma,  alphaL, nL,alphaR, nR, sigmaG):
        return expExp_pol2_turnOnPol1(x, normBkg, B, C,b,c, aa, f) + normSig*(fraction_dscb*dscb(x, mean, sigma,  alphaL, nL,alphaR, nR) + (1-fraction_dscb)*gaussianN(x, mean, sigmaG))