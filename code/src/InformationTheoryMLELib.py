import numpy as np
from scipy.optimize import minimize

def MLE(LL,data,x0,method):
    """
    Computes the maximum value for the log-likelihood function and returns the arguments of the parameters.
    Reference: https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/4a8de32565ebdefbb7963b4ebda904b2_MIT18_05S14_Reading10b.pdf
    Reference (scipy optimization): https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    Parameters:
    LL:  function
        log-likelihood as a function of the parameters and the data. Use log of parameters for smoother convergence
    data:  array
        array with data points.
    x0:  1d array
        set of initial points on optimization
    method: string
        optimization method.
        - ‘Nelder-Mead’
        - ‘BFGS’
        etc

    Returns:
    params:  array
        set of the argument of maximized LL
    """
    res = minimize(fun=lambda log_params, data: -LL(log_params, data),x0=x0, args=(data,), method=method)
    return(np.exp(res.x))
