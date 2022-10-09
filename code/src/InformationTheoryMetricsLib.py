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

    sum_log_distances_to_k_neighbor = np.sum(np.log(2*distances_to_k_neighbor))
    h = -digamma(k) + digamma(n) + log_c_d_p + (1.*d/n) * sum_log_distances_to_k_neighbor

    return(h)

def get_mutual_information(x,y,p=2,k=1,method='both'):
    """
    Computes the estimated mutual information for two continuous distributions following the
    Kraskov, Stogbauer and Grassberger (KSG) method (PhysRevE.69.066138 - 2014)
    Reference: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138
    Method 1: Balls Equation 23
    Method 2: Rectangle Equation 30

    Parameters:
    x:  array-like
        points in the distribution
    y:  array-like
        points in the distribution
    p:  scalar (p>=1)
        p value of the p-norm.
            p = 1, Manhattan
            p = 2, Eucledian
            p = np.inf, Max
    k:  scalar
        k neighbor
    method: 'balls','rectangles' or 'both'

    Returns:
    h:  scalar
        Approximated entropy for the distribution
    """

    # NOTE: do x,y need to be the same size?

    if len(x.shape) == 2:
        nx, dx = x.shape
    if len(y.shape) == 2:
        ny, dy = y.shape
    else:
        raise Exception("Input data set should be a 2-dimensional array")

    if p>=1:
        log_c_d_p = get_log_volume_ball_d_p(1, dx, p)
    else:
        #raise Exception("Input p must be p>=1")
        return('Nope')

    xy = np.c_[x,y]

    tree_x = KDTree(x)
    tree_y = KDTree(y)
    tree_xy = KDTree(xy)

    all_distances, indices = tree_xy.query(xy, k + 1, p=p)
    radii = all_distances[:,-1]

    lnx = np.zeros(nx)
    lny = np.zeros(ny)

    for i_dim in range(nx):
        lnx[i_dim] = len(tree_x.query_ball_point(tree_x.data[i_dim], r=radii[i_dim], p=p)) - 1
        lny[i_dim] = len(tree_y.query_ball_point(tree_y.data[i_dim], r=radii[i_dim], p=p)) - 1

    mi_balls = digamma(k) - np.mean(digamma(lnx+1) + digamma(lny+1)) + digamma(nx)
    mi_rectangles = digamma(k) -1./k - np.mean(digamma(lnx) + digamma(lny)) + digamma(nx)

    if method == 'balls':
        return(mi_balls)
    if method == 'rectangles':
        return(mi_rectangles)
    if method == 'both':
        return(np.mean(mi_balls+mi_rectangles))
    else:
        #raise Exception("Method should be 'balls','rectangles' or 'both'")
        return('Nope')




x = np.random.randn(10,2)
y = np.random.randn(10,2)

print(get_mutual_information(x,y,p=2,k=1,method='balls'))
print(get_mutual_information(x,y,p=2,k=1,method='rectangles'))
