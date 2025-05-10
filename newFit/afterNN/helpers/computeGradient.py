import numpy as np
def numerical_gradient(func, x, params, eps=1e-5):
    """
    Compute the gradient of `func(x, *params)` w.r.t. `params` using central finite differences.
    
    Parameters:
        func   : callable, function to differentiate (e.g., myBkgSignalFunctions[key])
        x      : array-like, input values to evaluate the function on
        params : array-like, parameters at which to evaluate the gradient
        eps    : float, step size for finite difference
    
    Returns:
        grad : np.ndarray, gradient array of shape (len(params), len(x))
    """
    params = np.array(params)
    grad = np.zeros((len(params), len(x)))
    
    for i in range(len(params)):
        p_up = params.copy()
        p_down = params.copy()
        p_up[i] += eps
        p_down[i] -= eps
        f_up = func(x, *p_up)
        f_down = func(x, *p_down)
        grad[i] = (f_up - f_down) / (2 * eps)
    
    return grad
