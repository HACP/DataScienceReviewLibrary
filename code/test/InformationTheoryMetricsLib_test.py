from src.InformationTheoryMetricsLib import *
import pytest

def test_get_entropy_valid_p():
    x = np.random.normal(size=(100,1))
    p = -1
    assert get_entropy(x,p) == 'Nope'

def test_get_log_volume_ball_d_p():
    """
    Test extreme cases of the logarithm of the volume of the d-dimensional ball
    with p-norm.
    """
    D = 1
    d1 = 1
    d2 = 2
    p1 = 1
    p2 = 2
    d3,p3 = 3, np.inf

    logV1_1 = get_log_volume_ball_d_p(D, d1, p1) # volume of 1-dimensional ball, Manhattan. V1_1(D) = D
    logV2_2 = get_log_volume_ball_d_p(D, d2, p2) # volume of 2-dimensional ball, Eucledian. V2_2(D) = pi*D^2/4
    logV3_inf = get_log_volume_ball_d_p(D, d3, p3) # volume of 3-dimensional ball, Max norm. V3_inf(D) = D^3
    assert np.linalg.norm(np.array([logV1_1, logV2_2, logV3_inf]) - np.array([0., np.log(np.pi/4),0])) < 0.01

def test_get_entropy_gaussian():
    """
    Test entropy for a continous distribution - a gaussian and compare against
    analytical result
    """
    # https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
    x = np.random.normal(size=(100000,1))
    assert abs(0.5*np.log(2*np.pi) + 0.5 - get_entropy(x,p=2,k=5))<0.01

def test_get_mutual_information_multivariate_gaussian():
    """
    Test mutual information for two continous distributions - two gaussians distributions and compare against
    analytical result
    """
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]
    NN = 50000
    x, y = np.random.multivariate_normal(mean, cov, NN).T
    assert abs(0 - get_mutual_information_continuous(x.reshape(NN,1),y.reshape(NN,1),p=np.inf,k=10,method='balls')) < 0.01

def test_get_mutual_information_mixed_multivariate_gaussian():
    """
    Test mutual information for two continous distributions - two gaussians distributions and compare against
    analytical result
    """
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]
    NN = 50000
    x, y = np.random.multivariate_normal(mean, cov, NN).T
    assert abs(0 - get_mutual_information_mixed(x.reshape(NN,1),y.reshape(NN,1),p=np.inf,k=10)) < 0.01

def test_get_mutual_information_mixed_discrete():
    """
    Test mutual information as self-information or entropy of a two valued discrete
    distribution with p = 0.5
    """
    X = np.random.choice([0,1],size=(10000,1))
    assert abs(np.log(2) - get_mutual_information_mixed(X,X,p=np.inf,k=5)) < 0.001
