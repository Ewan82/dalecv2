import numpy as np
#import ad


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
    phi_onset = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((dC.D[x] - d_onset + offset) / dC.radconv)*dC.radconv /\
                releasecoeff)**2)
    return phi_onset

    
def phi_fall(d_fall, crfall, clspan, dC, x):
    """Leaf fall function (controls foliar to litter carbon transfer) takes 
    d_fall value, crfall value, clspan value, a dataClass (dC) and a time step
    x and returns a value for phi_fall.
    """
    releasecoeff = np.sqrt(2.)*crfall / 2.
    magcoeff = (np.log(clspan) - np.log(clspan -1.)) / 2.
    offset = fitpolynomial(clspan, releasecoeff)
    phi_fall = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                np.sin((dC.D[x] - d_fall + offset) / dC.radconv)*dC.radconv / \
                releasecoeff)**2)
    return phi_fall
    

def dalecv2(clab, cf, cr, cw, cl, cs, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
            p12, p13, p14, p15, p16, p27, dC, x):
            return clab
            
    
    
    
    