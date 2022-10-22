from src.InformationTheoryMLELib import *
import pytest
import numpy as np

def test_get_MLE_gaussian():
    # performance of optimization methods can vary -
    data = np.random.normal(100,1,size=100000)

    def LL_gaussian(log_params,data):
        n = len(data)
        mu = np.exp(log_params[0])
        sigma = np.exp(log_params[1])
        return -(n/2)*np.log(2*np.pi)-n*np.log(sigma)-(1/(2*sigma*sigma))*np.sum((data-mu)*(data-mu))

    assert 1.*np.isclose(MLE(LL_gaussian,data,x0=np.array([10,10]),method='CG'),np.array([100,1]),0.01).sum() == 2

#Bill Withers - Lean on me
