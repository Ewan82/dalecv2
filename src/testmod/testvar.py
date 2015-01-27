"""Tests for the functions in the fourdvar module.
"""
import numpy as np
from model import oregondata as dC
from model import fourdvar as var
from model import fourdvarsum as varsm
import matplotlib.pyplot as plt


def test_costfn(alph=1e-9):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10,147)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = var.gradcost(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = var.cost(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = var.cost(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    print (jalph-j) / (np.dot(alph*h, gradj))
    assert (jalph-j) / (np.dot(alph*h, gradj)) < 1.0001


def test_costsumfn(alph=1e-9):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10,147)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = varsm.gradcostsum(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = varsm.costsum(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = varsm.costsum(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    print (jalph-j) / (np.dot(alph*h, gradj))
    assert (jalph-j) / (np.dot(alph*h, gradj)) < 1.0001
    
    
def test_cost(alph=1e-8, vect=0):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(300)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = var.gradcost(d.pvals, obdict, oberrdict, d, 0, 300)
    if vect == True:
        h = d.pvals*(np.linalg.norm(d.pvals))**(-1)
    else:
        h = gradj*(np.linalg.norm(gradj))**(-1)
    j = var.cost(d.pvals, obdict, oberrdict, d, 0, 300)
    jalph = var.cost(d.pvals + alph*h, obdict, oberrdict, d, 0, 300)
    print jalph
    print j
    print np.dot(alph*h, gradj)
    return (jalph-j) / (np.dot(alph*h, gradj))
    
    
def test_costsum(alph=1e-8):
    """Test for cost and gradcost functions.
    """
    d = dC.dalecData(10,147)
    obdict, oberrdict = d.assimilation_obs('nee')
    gradj = varsm.gradcostsum(d.pvals, obdict, oberrdict, d, 0, 10)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = varsm.costsum(d.pvals, obdict, oberrdict, d, 0, 10)
    jalph = varsm.costsum(d.pvals + alph*h, obdict, oberrdict, d, 0, 10)
    return (jalph-j) / (np.dot(alph*h, gradj)) 
    
    
def plotcost():
    power=np.arange(0,7,1)
    xlist = [10**(-x) for x in power]
    tstlist = [abs(1-test_cost(x, 1)) for x in xlist]
    plt.semilogx(xlist, tstlist)
    plt.xlabel('alpha')
    plt.ylabel('grad test function')
    print tstlist
    plt.show()