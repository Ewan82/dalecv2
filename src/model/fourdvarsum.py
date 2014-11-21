"""4DVAR scheme for dalecv2.
"""
import numpy as np
import scipy.optimize as spop
import model as m
import observations as obs


def obscostsum(obdict, oberrdict):
    """Function returning list of observations and a list of their 
    corresponding error values. Takes observation dictionary and an observation
    error dictionary.
    """
    yoblist = []
    yerrlist = []
    for x in xrange(len(obdict.values()[0])):
        temp = []
        temperr = []
        for ob in obdict.iterkeys():
            if np.isnan(obdict[ob][x])!=True:
                temp.append(obdict[ob][x])
                temperr.append(oberrdict[ob+'_err'][x])
        if len(temp)!=0:
            yoblist.append(temp)
            yerrlist.append(temperr)
        else:
            continue
    return yoblist, yerrlist
    
    
def hxcostsum(pvallist, obdict, dC):
    """Function returning a list of observation values as predicted by the 
    DALEC model. Takes a list of model values (pvallist), an observation 
    dictionary and a dataClass (dC).
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai}
    hx = []
    for x in xrange(len(obdict.values()[0])):
        temp = []
        for ob in obdict.iterkeys():
            if np.isnan(obdict[ob][x])!=True:
                temp.append(modobdict[ob](pvallist[x], dC,dC.timestep[x]))
        if len(temp)!=0:
            hx.append(temp)
        else:
            continue
    return hx
    
    
def rmatlist(yerr):
    """Returns observation error covariance matrix given a list of observation
    error values
    """
    r = []
    for x in xrange(len(yerr)):
        r.append(yerr[x]*np.eye(len(yerr[x])))
    return r
    
    
def hmatlist(pvallist, obdict, matlist, dC):
    """Returns a list of observation values as predicted by DALEC (hx) and a 
    linearzied observation error covariance matrix (hmat). Takes a list of 
    model values (pvallist), a observation dictionary, a list of linearized 
    models (matlist) and a dataClass (dC).
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai': obs.lai}
    hx = []
    hmat = []
    for x in xrange(len(obdict.values()[0])):
        temp = []
        temphmat = []
        for ob in obdict.iterkeys():
            if np.isnan(obdict[ob][x])!=True:
                temp.append(modobdict[ob](pvallist[x], dC,dC.timestep[x]))
                temphmat.append(obs.linob(ob, pvallist[x], dC, dC.timestep[x]))
        if len(temp)!=0:
            hx.append(temp)
        if len(temphmat) != 0.:
            hmat.append(np.dot(m.mfacadj(matlist, x-1),np.vstack(temphmat).T))
        else:
            continue
    return hx, hmat 
    
    
def costsum(pvals, obdict, oberrdict, dC, start, fin):
    """4DVAR cost function to be minimized. Takes an initial state (pvals), an
    observation dictionary, observation error dictionary, a dataClass and a
    start and finish time step.
    """
    pvallist = m.mod_list(pvals, dC, start, fin)
    yoblist, yerrlist = obscostsum(obdict, oberrdict)
    rlist = rmatlist(yerrlist)
    hx = hxcostsum(pvallist, obdict, dC)
    obcost=np.ones(len(yerrlist))*-9999.
    for x in xrange(len(yoblist)):
        obcost[x] = np.dot(np.dot((np.array(yoblist[x])-np.array(hx[x])),\
                    np.linalg.inv(rlist[x])),(np.array(yoblist[x])\
                    -np.array(hx[x])).T)
    #modcost =  np.dot(np.dot((pvals-dC.pvals),np.linalg.inv(dC.B)),\
    #                  (pvals-dC.pvals).T)
    cost = 0.5*np.sum(obcost) #+ 0.5*modcost
    return cost
#Try changing this for M.tH.t to see if this works better?
    
    
def gradcostsum(pvals, obdict, oberrdict, dC, start, fin):
    """Gradient of 4DVAR cost fn to be passed to optimization routine. Takes an
    initial state (pvals), an obs dictionary, an obs error dictionary, a 
    dataClass and a start and finish time step.
    """
    pvallist, matlist = m.linmod_list(pvals, dC, start, fin)
    yoblist, yerrlist = obscostsum(obdict, oberrdict)
    rlist = rmatlist(yerrlist)
    hx, hmatrix = hmatlist(pvallist, obdict, matlist, dC)
    obcost=[-9999.]*len(yoblist)
    for x in xrange(len(yoblist)):
        obcost[x] = np.dot(hmatrix[x], np.dot(np.linalg.inv(rlist[x]),\
                    (np.array(yoblist[x])-np.array(hx[x])).T))
    #modcost =  np.dot(np.linalg.inv(dC.B),(pvals-dC.pvals).T)
    obcostsum=obcost[0]
    for x in xrange(1,len(yerrlist)):
        obcostsum=obcostsum+obcost[x]
    gradcost =  - obcostsum.T #+ modcost
    return gradcost
    
    
def findminsum(pvals, obdict, oberrdict, dC, start, fin, meth='L-BFGS-B',\
            bnds='+ve', fprime=gradcostsum):
    """Function which minimizes 4DVAR cost fn. Takes an initial state (pvals),
    an obs dictionary, an obs error dictionary, a dataClass and a start and 
    finish time step.
    """
    if bnds == 'strict':
        bnds=((10,1000),(10,1000),(10,1000),(100,1e5),(10,1000),(100,2e5),\
              (1e-5,1e-2),(0.3,0.7),(0.01,0.5),(0.01,0.5),(1.1,10.),\
              (2.5e-5,1e-3),(1e-4,1e-2),(1e-4,1e-2),(1e-7,1e-3),(0.018,0.08),\
              (10,100),(1,365),(0.01,0.5),(10,100),(1,365),(10,100),(10,400))
    else:
        bnds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None),(0,None),(1.1,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None))
    findmin = spop.minimize(costsum, pvals, args=(obdict, oberrdict, dC, start,\
              fin,), method=meth, jac=fprime, bounds=bnds)
    return findmin
    
    
def findminglob(pvals, obdict, oberrdict, dC, start, fin, it=100,\
                bnds='strict', stpsize=0.5, temp=1.):
    """Function which minimizes 4DVAR cost fn. Takes an initial state (pvals),
    an obs dictionary, an obs error dictionary, a dataClass and a start and 
    finish time step.
    """
    if bnds == 'strict':
        bnds = ((10,1000),(10,1000),(10,1000),(100,1e5),(10,1000),(100,2e5),\
               (1e-5,1e-2),(0.3,0.7),(0.01,0.5),(0.01,0.5),(1.1,10.),\
               (2.5e-5,1e-3),(1e-4,1e-2),(1e-4,1e-2),(1e-7,1e-3),(0.018,0.08),\
               (10,100),(1,365),(0.01,0.5),(10,100),(1,365),(10,100),(10,400))
    else:
        bnds = bnds
    findmin = spop.basinhopping(costsum, pvals, niter=it, minimizer_kwargs={\
              'method': 'L-BFGS-B', 'args':(obdict, oberrdict, dC, start,fin),\
              'bounds':bnds}, stepsize=stpsize, T=temp)
    return findmin
