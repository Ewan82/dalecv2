"""4DVAR scheme for dalecv2.
"""
import numpy as np
import scipy.optimize as spop
import model as m
import observations as obs


def obscost(obdict, oberrdict):
    """Function returning list of observations and a list of their 
    corresponding error values. Takes observation dictionary and an observation
    error dictionary.
    """
    yoblist = np.array([])
    yerrlist = np.array([])
    for x in xrange(len(obdict.values()[0])):
        for ob in obdict.iterkeys():
            if np.isnan(obdict[ob][x])!=True:
                yoblist = np.append(yoblist, obdict[ob][x])
                yerrlist = np.append(yerrlist, oberrdict[ob+'_err'][x])
    return yoblist, yerrlist
              
                
def hxcost(pvallist, obdict, dC):
    """Function returning a list of observation values as predicted by the 
    DALEC model. Takes a list of model values (pvallist), an observation 
    dictionary and a dataClass (dC).
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai}
    hx = np.array([])
    for x in xrange(len(obdict.values()[0])):
        for ob in obdict.iterkeys():
            if np.isnan(obdict[ob][x])!=True:
                hx = np.append(hx, modobdict[ob](pvallist[x], dC,x))
    return hx
             
              
def rmat(yerr):
    """Returns observation error covariance matrix given a list of observation
    error values
    """
    r = yerr*np.eye(len(yerr))
    return r
    
    
def hmat(pvallist, obdict, matlist, dC):
    """Returns a list of observation values as predicted by DALEC (hx) and a 
    linearzied observation error covariance matrix (hmat). Takes a list of 
    model values (pvallist), a observation dictionary, a list of linearized 
    models (matlist) and a dataClass (dC).
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai': obs.lai}
    hx = np.array([])
    hmat = np.array([])
    for x in xrange(len(obdict.values()[0])):
        temp = []
        for ob in obdict.iterkeys():
            if np.isnan(obdict[ob][x])!=True:
                hx = np.append(hx, modobdict[ob](pvallist[x], dC,x))
                temp.append([obs.linob(ob, pvallist[x], dC, x)])
        if len(temp) != 0.:
            hmat = np.append(hmat, np.dot(np.vstack(temp),\
                             m.mfac(matlist, x-1)))
        else:
            continue
    hmat = np.reshape(hmat, (len(hmat)/23,23))
    return hx, hmat  
    
    
def cost(pvals, obdict, oberrdict, dC, start, fin):
    """4DVAR cost function to be minimized. Takes an initial state (pvals), an
    observation dictionary, observation error dictionary, a dataClass and a
    start and finish time step.
    """
    pvallist = m.mod_list(pvals, dC, start, fin)
    yoblist, yerrlist = obscost(obdict, oberrdict)
    rmatrix = rmat(yerrlist)
    hx = hxcost(pvallist, obdict, dC)
    obcost = np.dot(np.dot((yoblist-hx),np.linalg.inv(rmatrix)),(yoblist-hx).T)
    #modcost =  np.dot(np.dot((pvals-dC.pvals),np.linalg.inv(dC.B)),\
    #                  (pvals-dC.pvals).T)
    cost = 0.5*obcost #+ 0.5*modcost
    return cost




def gradcost(pvals, obdict, oberrdict, dC, start, fin):
    """Gradient of 4DVAR cost fn to be passed to optimization routine. Takes an
    initial state (pvals), an obs dictionary, an obs error dictionary, a 
    dataClass and a start and finish time step.
    """
    pvallist, matlist = m.linmod_list(pvals, dC, start, fin)
    yoblist, yerrlist = obscost(obdict, oberrdict)
    rmatrix = rmat(yerrlist)
    hx, hmatrix = hmat(pvallist, obdict, matlist, dC)
    obcost = np.dot(hmatrix.T, np.dot(np.linalg.inv(rmatrix), (yoblist-hx).T))
    #modcost =  np.dot(np.linalg.inv(dC.B),(pvals-dC.pvals).T)
    gradcost =  - obcost #+ modcost
    return gradcost
    
    
def findmin(pvals, obdict, oberrdict, dC, start, fin, meth='L-BFGS-B',\
            bnds='+ve', fprime=gradcost):
    """Function which minimizes 4DVAR cost fn. Takes an initial state (pvals),
    an obs dictionary, an obs error dictionary, a dataClass and a start and 
    finish time step.
    """
    if bnds == 'strict':
        bnds=((10,1000),(10,1000),(10,1000),(100,1e5),(10,1000),(100,2e5),\
              (1e-5,1e-2),(0.3,0.7),(0.01,0.5),(0.01,0.5),(1.0001,10.),\
              (2.5e-5,1e-3),(1e-4,1e-2),(1e-4,1e-2),(1e-7,1e-3),(0.018,0.08),\
              (10,100),(1,365),(0.01,0.5),(10,100),(1,365),(10,100),(10,400))
    else:
        bnds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None),(0,None),(1.0001,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None))
    findmin = spop.minimize(cost, pvals, args=(obdict, oberrdict, dC, start,\
              fin,), method=meth, jac=fprime, bounds=bnds,\
              options={'gtol': 1e-6, 'disp': True, 'iprint':2})
    return findmin
    
    
def findminglob(pvals, obdict, oberrdict, dC, start, fin, it=100,\
                bnds='strict', stpsize=0.5, temp=1., fprime=gradcost):
    """Function which minimizes 4DVAR cost fn. Takes an initial state (pvals),
    an obs dictionary, an obs error dictionary, a dataClass and a start and 
    finish time step.
    """
    if bnds == 'strict':
        bnds = ((10,1000),(10,1000),(10,1000),(100,1e5),(10,1000),(100,2e5),\
               (1e-5,1e-2),(0.3,0.7),(0.01,0.5),(0.01,0.5),(1.0001,10.),\
               (2.5e-5,1e-3),(1e-4,1e-2),(1e-4,1e-2),(1e-7,1e-3),(0.018,0.08),\
               (10,100),(1,365),(0.01,0.5),(10,100),(1,365),(10,100),(10,400))
    else:
        bnds = bnds
    findmin = spop.basinhopping(cost, pvals, niter=it, minimizer_kwargs={\
              'method': 'L-BFGS-B', 'args':(obdict, oberrdict, dC, start,fin),\
              'bounds':bnds, 'jac':fprime}, stepsize=stpsize, T=temp)
    return findmin