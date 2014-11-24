"""Tests for the functions in the fourdvar module.
"""
import numpy as np
from model import oregondata as dC
from model import fourdvar as var
from model import fourdvarsum as varsm
import matplotlib.pyplot as plt


def test_costfn(alph=1e-8):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = var.gradcost(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = var.cost(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = var.cost(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    print (jalph-j) / (np.dot(alph*h, gradj))
    assert (jalph-j) / (np.dot(alph*h, gradj)) < 1.0001


def test_costsumfn(alph=1e-8):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = varsm.gradcostsum(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = varsm.costsum(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = varsm.costsum(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    print (jalph-j) / (np.dot(alph*h, gradj))
    assert (jalph-j) / (np.dot(alph*h, gradj)) < 1.0001
    
    
def test_cost(alph=1e-8):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10,147)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = var.gradcost(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = var.cost(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = var.cost(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    return (jalph-j) / (np.dot(alph*h, gradj))
    
    
def test_costsum(alph=1e-8):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10)
    obdict, oberrdict = d.assimilation_obs('gpp')
    gradj = varsm.gradcostsum(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = varsm.costsum(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = varsm.costsum(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    return (jalph-j) / (np.dot(alph*h, gradj)) 
    
    
def plotcost():
    power=np.arange(5,12,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_cost(x) for x in xlist]
    plt.plot(xlist, tstlist)
    plt.show()