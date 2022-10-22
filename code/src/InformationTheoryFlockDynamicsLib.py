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

def get_minimum_PSO(objective_function,search_area_center, search_area_scale, NP, v, dt, c1=0.1, c2=0.1, w=0.8, generate_plots=True, eps = 0.01, N_iterations=100):
    # https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf
    """
     Estimates minimum value and the arguments for a continuous non-linear function using the particle swarming
     optimization algorithm.
     Reference: https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf
                https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/

     Parameters:
     objective_function: object
         funtion to be minimized. It should be a numpy function.
     search_area_center: array
         center of the (square) area of interest
     search_area_scale: scalar
         length of the (square) area of interest
     NP: scalar
         number of particles
     v, dt: scalars
         velocity and delta t. It can be combined into vdt
     c1,c2,w: scalars
         PSO parameters (TODO: explore meaning and further details)
     generate_plots: boolean
         True: generates the plots for each iteration (very slow)
         False: produces only the min and the argument (faster)
     N_iterations: scalar
        number of iterations
     eps: scalar
         tolerance to change in the best value (TODO: improve early stopping)

     Returns:
     global_best_objective:  scalar
         minimum value of teh objective function
     global_best: array
         position of minimum value
     """

    # initializing positions and speeds of particles
    uP = np.random.uniform(-search_area_scale/2,search_area_scale/2.,size=(NP,2)) + search_area_center
    vP = v*np.random.randn(NP,2)
    # computing initial particle and global best values
    particle_best = uP
    particle_best_objective = objective_function(uP[:,0],uP[:,1])
    global_best = particle_best[particle_best_objective.argmin()]
    global_best_objective = particle_best_objective.min()

    for jj in range(N_iterations):
        # updating postions and speeds of particles
        r = np.random.rand(2)
        vP = w * vP + c1*r[0]*(particle_best - uP) + c2*r[1]*(global_best-uP)
        uP = uP + vP*dt

        # updating particle and global best values
        objective = objective_function(uP[:,0], uP[:,1])
        particle_best[objective <= particle_best_objective] = uP[objective <= particle_best_objective]
        particle_best_objective = np.array([particle_best_objective, objective]).min(axis=0)
        #global_best = particle_best[particle_best_objective.argmin()]
        if np.abs(global_best_objective-particle_best_objective.min()) < eps:
            print("Early Stop at " + str(jj) + " iterations")
            return(global_best_objective, global_best)
        else:
            global_best_objective = particle_best_objective.min()

        if generate_plots == True:
            fig, ax = plt.subplots(figsize=(10,10))
            CS = ax.contour(X, Y, Z)
            ax.clabel(CS, inline=True, fontsize=10)
            plt.imshow(Z, extent=[search_area_center[0]-search_area_scale/2,search_area_center[0]+search_area_scale/2,search_area_center[1]-search_area_scale/2,search_area_center[1]+search_area_scale/2], origin='lower', cmap='viridis', alpha=0.5)
            plt.colorbar()
            plt.scatter([X_min],[Y_min],marker='*',c='k',s=100)
            plt.scatter(*(zip(*uP)),c='green')
            plt.quiver(*(zip(*uP)),*(zip(*vP)) , color='green', width=0.005, angles='xy', scale_units='xy', scale=1)
            plt.show()
    if generate_plots == False:
        fig, ax = plt.subplots(figsize=(10,10))
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=True, fontsize=10)
        plt.imshow(Z, extent=[search_area_center[0]-search_area_scale/2,search_area_center[0]+search_area_scale/2,search_area_center[1]-search_area_scale/2,search_area_center[1]+search_area_scale/2], origin='lower', cmap='viridis', alpha=0.5)
        plt.colorbar()
        plt.scatter([X_min],[Y_min],marker='*',c='k',s=100)
        plt.scatter(*(zip(*uP)),c='green')
        plt.quiver(*(zip(*uP)),*(zip(*vP)) , color='green', width=0.005, angles='xy', scale_units='xy', scale=1)
        plt.show()

    return(global_best_objective, global_best)
