from src.InformationTheoryDimensionLib import *
import pytest

def test_get_volume_ball_d_p():
    d = 3
    p = 2
    R = 1
    assert np.abs(get_volume_ball_d_p(d=3,p=2,R=1) - 4*np.pi/3) < 0.01

def test_get_area_ball_dm1():
    d = 3
    R = 1
    assert np.abs(get_area_ball_dm1(dm1=d-1,R=1) - 4*np.pi) < 0.01

def test_MC_volume_unit_ball_d_p():
    d = 3
    p = 2
    R = 1
    assert np.abs(MC_volume_unit_ball_d_p(d,p,N_iterations = 10000) - (4*np.pi/3)/np.power(2,3)) < 0.01

# Nickelback - Photograph
