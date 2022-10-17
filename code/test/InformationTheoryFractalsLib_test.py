from src.InformationTheoryFractalsLib import *
import pytest

def test_get_fractal_dimension_henon():

    henon = get_henon_map(a=1.4,b=0.3,x0=0,y0=0,N_iterations = 10000)
    dim = get_fractal_dimension(fractal=henon,min_exp=-2, max_exp=0, delta_exp=0.1,N_trials=10)
    assert np.abs(dim - 1.261) < 0.01
