import numpy as np
from scipy.special import gamma, digamma
from scipy.spatial import KDTree

def get_log_volume_ball_d_p(D, d, p):
    """
    Computes the log of the volume of a d-ball of diameter D, (d - dimension) in a p-InformationTheoryMetricsLib
    Reference: https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Balls_in_Lp_norms
    NOTE: KSG definition of a unit ball MIGHT be diameter = 1.

    Parameters:
    D:  scalar
        diameter
    d:  scalar
        dimension
    p:  scalar
        p-norm value

    Returns:
    log_vol:    scalar
                logarithm of the d-ball, p-norm of diameter D
    """
    return d*np.log(gamma(1./p + 1)) + d*np.log(D) - np.log(gamma(1.*d/p + 1))

def get_entropy(x,p=2,k=1):
    """
    Computes the estimated entropy for a continuous distribution following the
    Kraskov, Stogbauer and Grassberger (KSG) method (PhysRevE.69.066138 - 2014)
    Reference: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138
    Equation 20

    Parameters:
    x:  array-like
        points in the distribution
    p:  scalar (p>=1)
        p value of the p-norm.
            p = 1, Manhattan
            p = 2, Eucledian
            p = np.inf, Max
    k:  scalar
        k neighbor

    Returns:
    h:  scalar
        Approximated entropy for the distribution
    """

    if len(x.shape) == 2:
        n, d = x.shape
    else:
        raise Exception("Input data set should be a 2-dimensional array")

    if p>=1:
        log_c_d_p = get_log_volume_ball_d_p(1, d, p)
    else:
        #raise Exception("Input p must be p>=1")
        return('Nope')

    tree = KDTree(x)
    all_distances, indices = tree.query(x, k + 1, p=p)
    distances_to_k_neighbor = all_distances[:,-1]
    print(all_distances)

    sum_log_distances_to_k_neighbor = np.sum(np.log(2*distances_to_k_neighbor))
    h = -digamma(k) + digamma(n) + log_c_d_p + (1.*d/n) * sum_log_distances_to_k_neighbor

    return(h)




#print(get_log_volume_ball_d_p(1,2,np.inf))
