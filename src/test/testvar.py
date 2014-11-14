"""Tests for the functions in the fourdvar module.
"""
import numpy as np
from model import data as dC
from model import fourdvar as var


def test_costfn(alph=1e-5):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(100)
    obdict, oberrdict = d.assimilation_obs('gpp')
    gradj = var.gradcost(d.pvals, obdict, oberrdict, d, 0, 100)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = var.cost(d.pvals, obdict, oberrdict, d, 0, 100)
    jalph = var.cost(d.pvals + alph*h, obdict, oberrdict, d, 0, 100)
    print (jalph-j) / (np.dot(alph*h, gradj))
    assert (jalph-j) / (np.dot(alph*h, gradj)) < 1.00001