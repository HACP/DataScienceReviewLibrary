from src.InformationTheoryClusteringLib import *
import pandas as pd
import pytest

def test_kMeans_multivariate_gaussian():
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

    df1_truth = pd.DataFrame(np.transpose([x1,y1]))
    df1_truth['true_label'] = 0
    df2_truth = pd.DataFrame(np.transpose([x2,y2]))
    df2_truth['true_label'] = 1
    df3_truth = pd.DataFrame(np.transpose([x3,y3]))
    df3_truth['true_label'] = 2

    df_truth = pd.concat([df1_truth,df2_truth,df3_truth])

    df_pred = get_kMeans(df, k=3)

    assert (df_truth['true_label'].astype('string') + '_' + df['label'].astype('string')).value_counts().reset_index()[0].tolist() == [500,500,500]
