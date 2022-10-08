from src.InformationTheoryMetricsLib import *
import pytest

def test_get_entropy_valid_p():
    x = []
    p = -1
    assert get_entropy(x,p) == 'Nope'

def test_get_log_volume_ball_d_p():
    D = 1
    d1 = 1
    d2 = 2
    p1 = 1
    p2 = 2
    d3,p3 = 3, np.inf

    logV1_1 = get_log_volume_ball_d_p(D, d1, p1) # volume of 1-dimensional ball, Manhattan. V1_1(D) = D
    logV2_2 = get_log_volume_ball_d_p(D, d2, p2) # volume of 2-dimensional ball, Eucledian. V2_2(D) = pi*D^2/4
    logV3_inf = get_log_volume_ball_d_p(D, d3, p3) # volume of 3-dimensional ball, Max norm. V3_inf(D) = D^d3
    assert np.linalg.norm(np.array([logV1_1, logV2_2, logV3_inf]) - np.array([0., np.log(np.pi/4),0])) < 0.01
