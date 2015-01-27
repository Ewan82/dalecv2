"""Functions for and related to the DALECV2 model.
"""
import numpy as np
from autodiff import function, gradient
from autodiff import escape as esc
import ad
import ad.admath as adm
import algopy as ag
import math
from algopy import UTPM
import scipy as sp

def fitpolynomial(ep, multfac):
    """Polynomial used to find phi_f and phi (offset terms used in phi_onset 
    and phi_fall), given an evaluation point for the polynomial and a 
    multiplication term.
    """
    cf = [2.359978471e-05, 0.000332730053021, 0.000901865258885,
          -0.005437736864888, -0.020836027517787, 0.126972018064287,
          -0.188459767342504]
    polyval=cf[0]*ep**6+cf[1]*ep**5+cf[2]*ep**4+cf[3]*ep**3+cf[4]*ep**2+\
            cf[5]*ep**1+cf[6]*ep**0
    phi = polyval*multfac
    return phi
    
@function   
def temp_term(Theta, D, tm, tr, rc, lat, ca, rt, pd, I, a2, a3, a4, a5, a6, 
              a7, a8, a9, a10):
    """Calculates the temperature exponent factor for carbon pool respirations
    given a value for Theta parameter, a dataClass (dC) and a time step x.
    """
    temp_term = np.exp(Theta*esc(tm))
    return temp_term
    
@function
def acm(cf, clma, ceff, D, tm, tr, rc, lat, ca, rt, pd, I, a2, a3, a4, a5, a6, 
        a7, a8, a9, a10):
    """Aggregated canopy model (ACM) function
    ------------------------------------------
    Takes a foliar carbon (cf) value, leaf mass per area (clma), canopy
    efficiency (ceff), a DataClass (dC) and a time step (x) and returns the 
    estimated value for Gross Primary Productivity (gpp) of the forest at that
    time.
    """
    L = cf / clma
    q = esc(a3) - esc(a4)
    gc = (abs(esc(pd)))**(esc(a10)) / (0.5*esc(tr) + esc(a6)*esc(rt))
    p = ((ceff*L) / gc)*np.exp(esc(a8)*esc(tm))
    ci = 0.5*(esc(ca) + q - p + np.sqrt((ca + q - p)**2 \
         -4*(esc(ca)*q - p*esc(a3))))
    E0 = (esc(a7)*L**2) / (L**2 + esc(a9))
    delta = -23.4*np.cos((360.*(esc(D) + 10) / 365.)*(np.pi/180.))*(np.pi/180.)
    s = 24*np.arccos(( - np.tan(esc(lat))*np.tan(delta))) / np.pi
    if s >= 24.:
        s = 24.
    elif s <= 0.:
        s = 0.
    else:
        s = s
    gpp = (E0*esc(I)*gc*(esc(ca) - ci))*(esc(a2)*s + esc(a5)) / \
          (E0*esc(I) + gc*(esc(ca) - ci))          
    return gpp

@function
def phi_onset(d_onset, cronset, D, rc):
    """Leaf onset function (controls labile to foliar carbon transfer) takes 
    d_onset value, cronset value, a dataClass (dC) and a time step x and
    returns a value for phi_onset.
    """
    releasecoeff = np.sqrt(2.)*cronset / 2.
    magcoeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
    offset = fitpolynomial(1+1e-3, releasecoeff)
    phi_onset = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((esc(D) - d_onset + offset) / esc(rc))*(esc(rc) /\
                releasecoeff))**2)
    return phi_onset

@function   
def phi_fall(d_fall, crfall, clspan, D, rc):
    """Leaf fall function (controls foliar to litter carbon transfer) takes 
    d_fall value, crfall value, clspan value, a dataClass (dC) and a time step
    x and returns a value for phi_fall.
    """
    releasecoeff = np.sqrt(2.)*crfall / 2.
    magcoeff = (np.log(clspan) - np.log(clspan -1.)) / 2.
    offset = fitpolynomial(clspan, releasecoeff)
    phi_fall = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((esc(D) - d_fall + offset) / esc(rc))*esc(rc) / \
                releasecoeff)**2)
    return phi_fall

             
def dalecv2(clab, cf, cr, cw, cl, cs, theta_min, f_auto, f_fol, f_roo, clspan,
            theta_woo, theta_roo, theta_lit, theta_som, Theta, ceff, d_onset, 
            f_lab, cronset, d_fall, crfall, clma, D, tm, tr, rc, lat, ca, rt,
            pd, I, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    """DALECV2 carbon balance model 
    -------------------------------
    evolves carbon pools to the next time step, taking the 6 carbon pool values
    and 17 parameters at time t and evolving them to time t+1. Function also 
    requires a dataClass (dC) and a time step x.
    """
    releasecoeff = np.sqrt(2.)*cronset / 2.
    magcoeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
    offset = fitpolynomial(1+1e-3, releasecoeff)
    phi_on = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((D - d_onset + offset) / rc)*(rc /\
                releasecoeff))**2)
                
    releasecoeff = np.sqrt(2.)*crfall / 2.
    magcoeff = (np.log(clspan) - np.log(clspan -1.)) / 2.
    offset = fitpolynomial(clspan, releasecoeff)
    phi_off = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((D - d_fall + offset) / rc)*rc / \
                releasecoeff)**2)
                
    L = cf / clma
    q = a3 - a4
    gc = (ag.absolute(pd))**(a10) / (0.5*tr + a6*rt)
    p = ((ceff*L) / gc)*np.exp(a8*tm)
    ci = 0.5*(ca + q - p + np.sqrt((ca + q - p)**2 \
         -4*(ca*q - p*a3)))
    E0 = (a7*L**2) / (L**2 + a9)
    delta = -23.4*ag.cos((360.*(D + 10) / 365.)*(np.pi/180.))*(np.pi/180.)
    tnum = - np.tan(lat)*ag.tan(delta)
    s = 24*(np.pi/2-tnum-tnum**3./6.-3.*tnum**5./40.) / np.pi
    #if s >= 24.:
    #    s = 24.
    #elif s <= 0.:
    #    s = 0.
    #else:
    #    s = s
    gpp = (E0*I*gc*(ca - ci))*(a2*s + a5) / \
          (E0*I + gc*(ca - ci))  
          
    temp = np.exp(Theta*tm)
    
    clab2 = (1 - phi_on)*clab + (1-f_auto)*(1-f_fol)*f_lab*gpp
    #cf2 = (1 - phi_off)*cf + phi_on*clab + (1-f_auto)*f_fol*gpp
    #cr2 = (1 - theta_roo)*cr + (1-f_auto)*(1-f_fol)*(1-f_lab)*f_roo*gpp
    #cw2 = (1 - theta_woo)*cw + (1-f_auto)*(1-f_fol)*(1-f_lab)*(1-f_roo)*gpp
    #cl2 = (1-(theta_lit+theta_min)*temp)*cl + theta_roo*cr + phi_off*cf
    #cs2 = (1 - theta_som*temp)*cs + theta_woo*cw + theta_min*temp*cl
    return clab2
         

@gradient(wrt='clab, cf, cr, cw, cl, cs, theta_min, f_auto, f_fol, f_roo, clspan,\
            theta_woo, theta_roo, theta_lit, theta_som, Theta, ceff, d_onset,\
            f_lab, cronset, d_fall, crfall, clma')   
def cf(clab, cf, cr, cw, cl, cs, theta_min, f_auto, f_fol, f_roo, clspan,
            theta_woo, theta_roo, theta_lit, theta_som, Theta, ceff, d_onset, 
            f_lab, cronset, d_fall, crfall, clma, D, tm, tr, rc, lat, ca, rt,
            pd, I, a2, a3, a4, a5, a6, a7, a8, a9, a10):

    releasecoeff = np.sqrt(2.)*cronset / 2.
    magcoeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
    offset = fitpolynomial(1+1e-3, releasecoeff)
    phi_on = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((esc(D) - d_onset + offset) / esc(rc))*(esc(rc) /\
                releasecoeff))**2)
                
    releasecoeff = np.sqrt(2.)*crfall / 2.
    magcoeff = (np.log(clspan) - np.log(clspan -1.)) / 2.
    offset = fitpolynomial(clspan, releasecoeff)
    phi_off = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((esc(D) - d_fall + offset) / esc(rc))*esc(rc) / \
                releasecoeff)**2)
                
    L = cf / clma
    q = esc(a3) - esc(a4)
    gc = (abs(esc(pd)))**(esc(a10)) / (0.5*esc(tr) + esc(a6)*esc(rt))
    p = ((ceff*L) / gc)*np.exp(esc(a8)*esc(tm))
    ci = 0.5*(esc(ca) + q - p + np.sqrt((ca + q - p)**2 \
         -4*(esc(ca)*q - p*esc(a3))))
    E0 = (esc(a7)*L**2) / (L**2 + esc(a9))
    delta = -23.4*np.cos((360.*(esc(D) + 10) / 365.)*(np.pi/180.))*(np.pi/180.)
    s = 24*np.arccos(( - np.tan(esc(lat))*np.tan(delta))) / np.pi
    if s >= 24.:
        s = 24.
    elif s <= 0.:
        s = 0.
    else:
        s = s
    gpp = (E0*esc(I)*gc*(esc(ca) - ci))*(esc(a2)*s + esc(a5)) / \
          (E0*esc(I) + gc*(esc(ca) - ci))  
          
    temp = np.exp(Theta*esc(tm))
    
    clab2 = (1 - phi_on)*clab + (1-f_auto)*(1-f_fol)*f_lab*gpp  
    return clab2

           
def dalecv2_input(p, rc, lat, ca, rt, pd, a2, a3, a4, a5, a6, a7, 
                  a8, a9, a10):
    """DALEC model passes a list or array of parameters to the function dalecv2
    takes, a list of parameters (p), a dataClass (dC) and a time step (x).
    append D, tm, tr, I.
    """
    dalecoutput = dalecv2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                          p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
                          p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], p[25], 
                          rc, lat, ca, rt, pd, p[26], a2, a3, a4, a5, a6, a7, a8, 
                          a9, a10)
    return dalecoutput
    
    
def dalecv2_input2(p):
    """DALEC model passes a list or array of parameters to the function dalecv2
    takes, a list of parameters (p), a dataClass (dC) and a time step (x). 
    p[0] to p[22] dependent vars.
    """
    dalecoutput = dalecv2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                          p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
                          p[17], p[18], p[19], p[20], p[21], p[22], p[23], 
                          p[24], p[25],p[26], p[27], p[28], p[29], p[30], 
                          p[31], p[32], p[33], p[34], p[35], p[36], p[37], 
                          p[38], p[39], p[40])
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

