from src.InformationTheoryFlockDynamicsLib import *
import pytest

def test_get_periodic_conditions():
    r = np.array([0.3])
    L = 1
    assert np.linalg.norm(get_periodic_conditions(r+1.1,L) - np.array([0.4]))<0.001

#boygenius - souvenir
