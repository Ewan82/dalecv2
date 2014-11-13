import numpy as np
import scipy.optimize as spop
import model as m
import observations as obs


def obscost(obdict, oberrdict):
    yoblist = np.array([])
    yerrlist = np.array([])
    for x in xrange(len(obdict.values()[0])):
        for ob in obdict.iterkeys():
            if obdict[ob][x]!=-9999.:
                yoblist = np.append(yoblist, obdict[ob][x])
                yerrlist = np.append(yerrlist, oberrdict[ob+'_err'][x])
    return yoblist, yerrlist
              
                
def hxcost(pvallist, obdict, dC):
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai}
    hx = np.array([])
    for x in xrange(len(obdict.values()[0])):
        for ob in obdict.iterkeys():
            if obdict[ob][x]!=-9999.:
                hx = np.append(hx, modobdict[ob](pvallist[x], dC,\
                               dC.timestep[x]))
    return hx
              
              
def rmat(yerr):
    r = yerr*np.eye(len(yerr))
    return r
    
    
def hmat(pvallist, obdict, matlist, dC):
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai': obs.lai}
    hx = np.array([])
    hmat = np.array([])
    for x in xrange(len(obdict.values()[0])):
        temp = []
        for ob in obdict.iterkeys():
            if obdict[ob][x] != -9999.:
                hx = np.append(hx, modobdict[ob](pvallist[x], dC,\
                               dC.timestep[x]))
                temp.append([obs.linob(ob, pvallist[x], dC, dC.timestep[x])])
        if len(temp) != 0.:
            hmat = np.append(hmat, np.dot(np.vstack(temp),\
                             m.mfac(matlist, x-1)))
        else:
            continue
    hmat = np.reshape(hmat, (len(hmat)/23,23))
    return hx, hmat   
    
    
def cost(pvals, obdict, oberrdict, dC, start, fin):
    pvallist = m.mod_list(pvals, dC, start, fin)
    yoblist, yerrlist = obscost(obdict, oberrdict)
    rmatrix = rmat(yerrlist)
    hx = hxcost(pvallist, obdict, dC)
    obcost = np.dot(np.dot((yoblist-hx),rmatrix),(yoblist-hx).T)
    modcost =  np.dot(np.dot((pvals-dC.pvals),dC.B),(pvals-dC.pvals).T)
    cost = 0.5*modcost + 0.5*obcost
    return cost


def gradcost(pvals, obdict, oberrdict, dC, start, fin):
    pvallist = m.mod_list(pvals, dC, start, fin)
    matlist = m.linmod_list(pvals, dC, start, fin)
    yoblist, yerrlist = obscost(obdict, oberrdict)
    rmatrix = rmat(yerrlist)
    hx, hmatrix = hmat(pvallist, obdict, matlist, dC)
    obcost = np.dot(np.dot(hmatrix.T,np.linalg.inv(rmatrix)),(yoblist-hx).T)
    modcost =  np.dot(np.linalg.inv(dC.B),(pvals-dC.pvals).T)
    gradcost = modcost - obcost
    return gradcost
    
    
def findmin(pvals, obdict, oberrdict, dC, start, fin):
    findmin = spop.minimize(cost, pvals, args=(obdict, oberrdict, dC, start,\
              fin,), method='L-BFGS-B', jac=gradcost, bounds=((0,None),(0,None),\
              (0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
              (0,None),(0.5,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
              (0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None)))
    return findmin