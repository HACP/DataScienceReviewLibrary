import numpy as np
import pandas as pd
import pylab as plt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import KDTree
from InformationTheoryMetricsLib import get_normalized_mutual_information_mixed as NMI

def get_kMeans(df, k, verbose=0):
    """
    Computes kMeans for an arbitrary (n,d) data set.

    Parameters:
    df: data frame (n,d) dimension
        n points in a R^d space

    k:  scalar
        k clusters

    verbose: integer
        verbose options:
        - 0: no extra information
        - 1: additional details about iterations

    Returns:
    df: data frame (n,d+1) dimension
        same input data frame with an additional column called 'label' with
        the cluster index
    """

    # initializing a random set of k centroids
    n, d = df.shape
    centroid0 = np.random.uniform(df.min().min(),df.max().max(),size=(k,d))
    lCentroids = [centroid0]

    # defining a working version of the input df.
    # computing the closest centroid index
    df_km = df.copy()
    df_km['closest_centroid'] = np.argmin(cdist(df, centroid0),axis=1)

    # repeat until there is no further change
    delta_centroids = 1
    iteration = 1
    while delta_centroids > 0:
        current_centroids = df_km['closest_centroid']
        # update centroid with mean value of input data per cluster
        centroid = np.array(df_km.groupby('closest_centroid').mean().reset_index()[np.arange(d)])
        lCentroids.append(centroid)
        df_km['closest_centroid'] = np.argmin(cdist(df, lCentroids[-1]),axis=1)
        delta_centroids = np.abs(df_km['closest_centroid'] - current_centroids).sum()

        if verbose == 1:
            print('Iteration Number ' + str(iteration))
            print(df_km['closest_centroid'].value_counts())

        iteration = iteration + 1

    df['label'] = df_km['closest_centroid']
    return(df)

def get_max_free_energy_clustering(df, N_C, T = 1000, k = 10, N_iterations = 1000):
    """
    Computes max free energy clustering for an arbitrary (n,d) data set.
    The algorithm maximized the "free energy" associated to the entropy and the similarity matrix.
    F = mean_s - T * I
    where mean_s is the mean value of the similarity measure, averaged over the probability P_I_C
    and I is the information content.
    We created a greedy way to find the minimum - too slow. This version leverages local probability
    Reference: https://www.princeton.edu/~wbialek/our_papers/slonim+al_05b.pdf

    Note: the algorithm can find local maxima - e.g. a true 3-cluster mapped with 2 computed clusters.
    Need to think a way to discourage high intra cluster distances/

    Parameters:
    df: data frame (n,d) dimension
        n points in a R^d space

    N_C:  scalar
        number of clusters

    T:  scalar
        Lagrange multiplier -

    k:  integer
        Number of k nearest neighbors to be flipped to mean probability

    N_iterations:   scalar
        Maximum number of iterations

    Returns:
    df: data frame (n,d+1) dimension
        same input data frame with an additional column called 'label' with
        the cluster index
    """

    N = len(df)
    def initialize_P_I_C(N):
        # initializing the array containing the probability that point I belongs to cluster C
        nn_P_I_C = np.random.rand(N,N_C)
        P_I_C = nn_P_I_C/np.sum(nn_P_I_C,axis=1).reshape(-1,1)
        return(P_I_C)

    tree = KDTree(df) # kd tree to evaluate k nearest neighbors

    # defining similarity metric based on the distance TO DO: include p as an option
    s = squareform(pdist(df))
    s = np.exp(-s)

    P_I_C = initialize_P_I_C(N)

    # starting the loop to maximze F
    Fmax = -10000000

    for _ in range(N_iterations):
        # selecting one point at random, and finding coordinates
        rand_sample_index = np.random.randint(N, size=1)
        df_sample = df.iloc[rand_sample_index]
        # finding k nearest neighbors
        dist, index_list = tree.query(df_sample,k)
        # defining working copy of P_I_C and assign mean value of P_I_C to the neighborhood
        P_I_C_test = P_I_C.copy()
        P_I_C_test[index_list] = np.mean(P_I_C_test[index_list],axis=1)

        # compute the average similarity among elements chosen independently out of a single cluster
        s_C = np.diag(np.matmul(np.matmul(np.transpose(P_I_C_test),s),P_I_C_test))
        # total probability of finding any element in cluster C
        P_C = np.sum(P_I_C_test,axis=0)*(1/N)
        # compute mean similarity across all data points and clusters
        mean_s = np.dot(s_C,P_C)
        # compute information carried by clusters
        I = np.sum(np.log(P_I_C_test*1/P_C)*P_I_C_test)/N
        # define free energy
        F = mean_s - T*I

        if F > Fmax:
        #print("Minimization",F)
            Fmax = F
            P_I_C = P_I_C_test

    df['label'] = np.argmax(P_I_C,axis=1)
    return(df)


mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
NN = 500
x1, y1 = np.random.multivariate_normal(mean1, cov1, NN).T
mean2 = [10, 0]
cov2 = [[1, 0], [0, 1]]
NN = 500
x2, y2 = np.random.multivariate_normal(mean2, cov2, NN).T
mean3 = [0, 10]
cov3 = [[1, 0], [0, 1]]
NN = 500
x3, y3 = np.random.multivariate_normal(mean3, cov3, NN).T

df1 = pd.DataFrame(np.transpose([x1,y1]))
df2 = pd.DataFrame(np.transpose([x2,y2]))
df3 = pd.DataFrame(np.transpose([x3,y3]))

df = pd.concat([df1,df2,df3])

print(df.head())
print(get_kMeans(df, k=3, verbose = 1))
print(get_max_free_energy_clustering(df, N_C=3))

df1_truth = pd.DataFrame(np.transpose([x1,y1]))
df1_truth['true_label'] = 0
df2_truth = pd.DataFrame(np.transpose([x2,y2]))
df2_truth['true_label'] = 1
df3_truth = pd.DataFrame(np.transpose([x3,y3]))
df3_truth['true_label'] = 2

df_truth = pd.concat([df1_truth,df2_truth,df3_truth])

print((df_truth['true_label'].astype('string') + '_' + df['label'].astype('string')).value_counts().reset_index()[0].tolist())
print(NMI(np.array(df_truth['true_label']).reshape(-1,1),np.array(df['label']).reshape(-1,1)))
