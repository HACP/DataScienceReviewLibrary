import numpy as np
import pylab as plt
from scipy.spatial import KDTree

def get_periodic_conditions(r,L):
    """
    Computes periodic boundaries for the square LxL. If a coordinates it's outside the [0,L] interval,
    it substracts multiples of L and reports the reminder (TODO: extend behavior in other dimension.)

    Parameters:
    r:  array-like
        points in the distribution
    L:  scalar
        Length of the boundaries

    Returns:
    r_periodic:  array_like
        points in the distribution with periodic conditions.
    """
    r_periodic = np.mod(r,L)
    return(r_periodic)

def get_update_theta(r,theta,eps,noise_mag):
    """
    Updates the velocity angles of all the particles based on the average of neighbors inside the ball of radius eps
    (TODO: extend behavior in other dimension.)

    Parameters:
    r:  array-like
        points in the distribution
    theta:  array
        velocity angles/direction of the particles
    eps: scalar
        radius of the neighborhood
    noise_mag: scalar
        magnitude of the noise added to the angles/directions

    Returns:
    theta:  array_like
        updated velocity angles/direction of the particles.
    """
    tree = KDTree(r)
    l_theta = []
    for item in r:
        neighbors = tree.query_ball_point(item,eps)
        l_theta.append(np.mean(theta[neighbors])+np.random.uniform()*2*np.pi*noise_mag)
    return(np.array(l_theta))

def get_update_r(r,theta,vdt,L):
    """
    Updates the position of all the particles based on the current position, the updated angles and (time*velocity)
    interval with periodic conditions. (TODO: extend behavior in other dimension.)

    Parameters:
    r:  array-like
        points in the distribution
    theta:  array
        velocity angles/direction of the particles
    vdt: scalar
        velocity*time interval
    L: scalar
        length of the boundary

    Returns:
    r:  array_like
        updated velocity angles/direction of the particles.
    """
    r = r + vdt*np.array([np.cos(theta),np.sin(theta)]).T
    return(get_periodic_conditions(r,L))

def get_update_iteration(r,theta,eps,noise_mag,vdt,L):
    """
    Updates the position and orientation of all the particles. (TODO: extend behavior in other dimension.)

    Parameters:
    r:  array-like
        points in the distribution
    theta:  array
        velocity angles/direction of the particles
    eps: scalar
        radius of the neighborhood
    noise_mag: scalar
        magnitude of the noise added to the angles/directions
    vdt: scalar
        velocity*time interval
    L: scalar
        length of the boundary

    Returns:
    r:  array_like
        updated velocity angles/direction of the particles.
    theta:  array_like
        updated velocity angles/direction of the particles
    """
    theta = get_update_theta(r,theta,eps,noise_mag)
    r = get_update_r(r,theta,vdt,L)
    return(r,theta)

def get_flock_dynamics(N,L,noise_mag,eps,vdt,N_iter):
    """
    Simulation of flock dynamics following Vicsek's model.
    - creates random distribution of points in the [0,L]X[0,L], uniform. Assign random direction theta
    - update theta and r
    iterate
    (TODO: check behavior in other dimension.)

    Parameters:
    N: scalar
        Number of particles
    eps: scalar
        radius of the neighborhood
    noise_mag: scalar
        magnitude of the noise added to the angles/directions
    vdt: scalar
        velocity*time interval
    L: scalar
        length of the boundary
    N_iter: scalar
        number of iterations

    Returns:
    """
    r = L*np.random.uniform(size=(N,2))
    theta = 2*np.pi*np.random.uniform(size=N)

    for iteration in range(N_iter):
        r,theta = get_update_iteration(r,theta,eps,noise_mag,vdt,L)

        plt.figure(figsize=(10,10))
        plt.scatter(*(zip(*r)))
        plt.xlim([0,L])
        plt.ylim([0,L])
        plt.axis('off')
        plt.savefig('flock_iteration_' + str(iteration).zfill(10) + '.png')
    os.system('convert -delay 10 -loop 0 flock_iteration_*.png flock_dynamics.gif')
