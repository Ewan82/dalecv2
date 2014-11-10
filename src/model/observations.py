"""Possible carbon balance observations shown as functions of the DALEC
variables and parameters to be used in data assimilation scheme.
"""

import model as m


def rec(pvals, dC, x):
    """Function calculates total ecosystem respiration (REC).
    """
    rec = pvals[7]*m.acm(pvals[1], pvals[16], dC, x) + (pvals[13]*pvals[4] +\
          pvals[14]*pvals[5])*m.temp_term(pvals[15], dC, x)
    return rec
    

def nee(pvals, dC, x):
    """Function calculates Net Ecosystem Exchange (NEE).
    """
    nee = rec(pvals, dC, x) - m.acm(pvals[1], pvals[16], dC, x)
    return nee
    
    
def lai(pvals, dC, x):
    """Fn calculates leaf area index (cf/clma).
    """
    lai = pvals[1] / pvals[22]
    return lai
    
    
def lf(pvals, dC, x):
    """Fn calulates litter fall.
    """
    lf = m.phi_fall(pvals[20], pvals[21], pvals[10], dC, x)*pvals[1]
    return lf