from scipy.special import gamma
import numpy as np

def get_volume_ball_d_p(d, p, R):
    """
    Computes the volume of the d-dimensional ball with radius R and p value.
    Reference: https://www.whitman.edu/documents/Academics/Mathematics/2014/jorgenmd.pdf

    Parameters:
    d:  scalar
        dimension
    p:  scalar (p>=1)
        p value of the p-norm.
            p = 1, Manhattan
            p = 2, Eucledian
            p = np.inf, Max
    R:  scalar
        Radius of d-ball R >= 0

    Returns:
    volume:  scalar
        exact value of the volume of n-ball
    """
    return np.power(2*gamma(1./p +1),d)*np.power(R,d)/gamma(1.*d/p+1)

def get_area_ball_dm1(dm1,R):
    """
     Computes the area of the (d-1)-dimensional ball with radius R .
     Reference: https://www.whitman.edu/documents/Academics/Mathematics/2014/jorgenmd.pdf

     Parameters:
     dm1:  scalar
         dimension - 1
     R:  scalar
         Radius of d-ball R >= 0

     Returns:
     area:  scalar
         exact value of the surface area of n-ball
     """

     return 2*np.power(np.pi,(dm1+1)/2.)*np.power(R,dm1)/gamma((dm1+1)/2.)

def MC_volume_unit_ball_d_p(d,p,N_iterations = 100):
    """
     Estimates the volume of the d-dimensional ball with radius R and p value using a Monte Carlo method.
     The idea is to find the ratio of uniformely distributed points in the d-cube that fall inside the
     unit d-ball.
     Reference:     # https://core.ac.uk/download/pdf/72843286.pdf

     Parameters:
     d:  scalar
         dimension
     p:  scalar (p>=1)
         p value of the p-norm.
         p = 1, Manhattan
         p = 2, Eucledian
         p = np.inf, Max
     N_iterations: scalar
        number of iterations in the MC

     Returns:
     ratio:  scalar
         ratio of volume n-ball and n-cube
     """     
    data = 2*(np.random.uniform(size=(N_iterations,d))-0.5)
    norm_data = np.linalg.norm(data,ord=p,axis=1)
    data_inside = np.sum((norm_data < 1)*1.)
    ratio = data_inside/N_iterations
    return(ratio)
