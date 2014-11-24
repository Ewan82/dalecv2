"""Functions for and related to the DALECV2 model.
"""
import numpy as np
import ad
import ad.admath as adm

def fitpolynomial(evalpoint, multfac):
    """Polynomial used to find phi_f and phi (offset terms used in phi_onset 
    and phi_fall), given an evaluation point for the polynomial and a 
    multiplication term.
    """
    polycoeffs = [2.359978471e-05, 0.000332730053021, 0.000901865258885,
                      -0.005437736864888, -0.020836027517787, 0.126972018064287,
                      -0.188459767342504]
    phi = np.polyval(polycoeffs, evalpoint)*multfac
    return phi
    
    
def temp_term(Theta, dC, x):
    """Calculates the temperature exponent factor for carbon pool respirations
    given a value for Theta parameter, a dataClass (dC) and a time step x.
    """
    temp_term = adm.exp(Theta*dC.t_mean[x])
    return temp_term
    

def acm(cf, clma, ceff, dC, x):
    """Aggregated canopy model (ACM) function
    ------------------------------------------
    Takes a foliar carbon (cf) value, leaf mass per area (clma), canopy
    efficiency (ceff), a DataClass (dC) and a time step (x) and returns the 
    estimated value for Gross Primary Productivity (gpp) of the forest at that
    time.
    """
    L = cf / clma
    q = dC.a3 - dC.a4
    gc = (abs(dC.phi_d))**(dC.a10) / (0.5*dC.t_range[x] + dC.a6*dC.R_tot)
    p = ((ceff*L) / gc)*np.exp(dC.a8*dC.t_max[x])
    ci = 0.5*(dC.ca + q - p + np.sqrt((dC.ca + q - p)**2 \
         -4*(dC.ca*q - p*dC.a3)))
    E0 = (dC.a7*L**2) / (L**2 + dC.a9)
    delta = -23.4*np.cos((360.*(dC.D[x] + 10) / 365.)*(np.pi/180.))*(np.pi/180.)
    s = 24*np.arccos(( - np.tan(dC.lat)*np.tan(delta))) / np.pi
    if s >= 24.:
        s = 24.
    elif s <= 0.:
        s = 0.
    else:
        s = s
    gpp = (E0*dC.I[x]*gc*(dC.ca - ci))*(dC.a2*s + dC.a5) / \
          (E0*dC.I[x] + gc*(dC.ca - ci))          
    return gpp


def phi_onset(d_onset, cronset, dC, x):
    """Leaf onset function (controls labile to foliar carbon transfer) takes 
    d_onset value, cronset value, a dataClass (dC) and a time step x and
    returns a value for phi_onset.
    """
    releasecoeff = np.sqrt(2.)*cronset / 2.
    magcoeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
    offset = fitpolynomial(1+1e-3, releasecoeff)
    phi_onset = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*adm.exp(-( \
                adm.sin((dC.D[x] - d_onset + offset) / dC.radconv)*(dC.radconv /\
                releasecoeff))**2)
    return phi_onset

    
def phi_fall(d_fall, crfall, clspan, dC, x):
    """Leaf fall function (controls foliar to litter carbon transfer) takes 
    d_fall value, crfall value, clspan value, a dataClass (dC) and a time step
    x and returns a value for phi_fall.
    """
    releasecoeff = np.sqrt(2.)*crfall / 2.
    magcoeff = (adm.log(clspan) - adm.log(clspan -1.)) / 2.
    offset = fitpolynomial(clspan, releasecoeff)
    phi_fall = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*adm.exp(-( \
                adm.sin((dC.D[x] - d_fall + offset) / dC.radconv)*dC.radconv / \
                releasecoeff)**2)
    return phi_fall
    
            
def dalecv2(clab, cf, cr, cw, cl, cs, theta_min, f_auto, f_fol, f_roo, clspan,
            theta_woo, theta_roo, theta_lit, theta_som, Theta, ceff, d_onset, 
            f_lab, cronset, d_fall, crfall, clma, dC, x):
    """DALECV2 carbon balance model 
    -------------------------------
    evolves carbon pools to the next time step, taking the 6 carbon pool values
    and 17 parameters at time t and evolving them to time t+1. Function also 
    requires a dataClass (dC) and a time step x.
    """
    clab2 = (1 - phi_onset(d_onset, cronset, dC, x))*clab + (1-f_auto)*(1-f_fol)\
            *f_lab*acm(cf, clma, ceff, dC, x)
    cf2 = (1 - phi_fall(d_fall, crfall, clspan, dC, x))*cf + \
          phi_onset(d_onset, cronset, dC, x)*clab + (1-f_auto)*f_fol*\
          acm(cf, clma, ceff, dC, x)
    cr2 = (1 - theta_roo)*cr + (1-f_auto)*(1-f_fol)*(1-f_lab)*f_roo*\
          acm(cf, clma, ceff, dC, x)
    cw2 = (1 - theta_woo)*cw + (1-f_auto)*(1-f_fol)*(1-f_lab)*(1-f_roo)*\
          acm(cf, clma, ceff, dC, x)
    cl2 = (1-(theta_lit+theta_min)*temp_term(Theta, dC, x))*cl + theta_roo*cr \
          +phi_fall(d_fall, crfall, clspan, dC, x)*cf
    cs2 = (1 - theta_som*temp_term(Theta, dC, x))*cs + theta_woo*cw + \
          theta_min*temp_term(Theta, dC, x)*cl
    return np.array((clab2, cf2, cr2, cw2, cl2, cs2, theta_min, f_auto, f_fol, 
           f_roo, clspan, theta_woo, theta_roo, theta_lit, theta_som, Theta, ceff, 
           d_onset, f_lab, cronset, d_fall, crfall, clma))
           
           
def dalecv2_input(p, dC, x):
    """DALEC model passes a list or array of parameters to the function dalecv2
    takes, a list of parameters (p), a dataClass (dC) and a time step (x).
    """
    dalecoutput = dalecv2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                          p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
                          p[17], p[18], p[19], p[20], p[21], p[22], dC, x)
    return dalecoutput
    
    
def lin_dalecv2(pvals, dC, x):
    """Linear DALEC model passes a list or array of parameters to the function 
    dalecv2 and returns a linearized model M for timestep xi. Takes, a list of 
    parameters (pvals), a dataClass (dC) and a time step (x).
    """
    p = ad.adnumber(pvals)
    dalecoutput = dalecv2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                          p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
                          p[17], p[18], p[19], p[20], p[21], p[22], dC, x)
    lin_model = ad.jacobian(dalecoutput, p)
    return dalecoutput, lin_model
    
    
def mod_list(pvals, dC, start, fin):
    """Creates an array of evolving model values using dalecv2_input function.
    Takes a list of initial param values, a dataClass (dC) and start and finish
    time.
    """
    mod_list = np.concatenate((np.array([pvals]),\
                               np.ones((fin - start, len(pvals)))*-9999.))
    for x in xrange(start, fin):
        mod_list[(x+1) - start] = dalecv2_input(mod_list[x-start], dC, x)
    return mod_list
    
    
def linmod_list(pvals, dC, start, fin):
    """Creates an array of linearized models (Mi's) taking a list of initial 
    param values, a dataClass (dC) and a start and finish time.
    """
    mod_list = np.concatenate((np.array([pvals]),\
                               np.ones((fin - start, len(pvals)))*-9999.))
    matlist = np.ones((fin - start,23,23))*-9999.
    for x in xrange(start, fin):
        mod_list[(x+1)-start], matlist[x-start] =\
                                          lin_dalecv2(mod_list[x-start], dC, x)
    return mod_list, matlist

    
def linmod_evolve(pvals, matlist, dC, start, fin):
    """evoles initial start (pvals) forward using given matrix list (matlist)
    of linearized models, also takes a dataClass (dC) and a start and finish 
    point.
    """    
    linmod_list = np.concatenate((np.array([pvals]),\
                                  np.ones((fin - start, len(pvals)))*-9999.))
    for x in xrange(start, fin):
        linmod_list[(x+1)-start] = np.dot(matlist[x-start],linmod_list[x-start])
    return linmod_list
    
    
def mfac(matlist, timestep):
    """matrix factorial function, takes a list of matrices and a time step,
    returns the matrix factoral.
    """
    if timestep==-1.:
        return np.eye(23)
    mat = matlist[0]
    for x in xrange(0,timestep):
        mat = np.dot(matlist[x+1], mat)
    return mat
      

def mfacadj(matlist, timestep):
    """matrix factoral fn for adjoint, takes a list of matrices and a timestep.
    """
    if timestep == -1:
        return np.eye(23)
    else:
        mat=matlist[0].T
        for x in xrange(0,timestep):
            mat=np.dot(mat, matlist[x+1].T)
    return mat
    
    
def linmod_evolvefac(pvals, matlist, dC, start, fin):
    """evoles initial start (pvals) forward using given matrix list (matlist)
    of linearized models, also takes a dataClass (dC) and a start and finish 
    point.
    """    
    linmod_list = np.concatenate((np.array([pvals]),\
                                  np.ones((fin - start, len(pvals)))*-9999.))
    for x in xrange(start, fin):
        linmod_list[(x+1)-start] = np.dot(mfac(matlist, x-start), pvals)
    return linmod_list
