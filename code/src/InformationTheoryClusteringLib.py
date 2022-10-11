import numpy as np
import pandas as pd
import pylab as plt
from scipy.spatial.distance import cdist, pdist, squareform
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

df1_truth = pd.DataFrame(np.transpose([x1,y1]))
df1_truth['true_label'] = 0
df2_truth = pd.DataFrame(np.transpose([x2,y2]))
df2_truth['true_label'] = 1
df3_truth = pd.DataFrame(np.transpose([x3,y3]))
df3_truth['true_label'] = 2

df_truth = pd.concat([df1_truth,df2_truth,df3_truth])

print((df_truth['true_label'].astype('string') + '_' + df['label'].astype('string')).value_counts().reset_index()[0].tolist())
print(NMI(np.array(df_truth['true_label']).reshape(-1,1),np.array(df['label']).reshape(-1,1)))
