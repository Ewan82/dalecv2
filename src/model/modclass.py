"""Dalecv2 model class takes a data class and then uses functions to run the
dalecv2 model.
"""
import numpy as np
import algopy


class dalecModel():
 
   
    def __init__(self, dataClass, timestep=0):
        """dataClass and timestep at which to run the dalecv2 model.
        """        
        self.dC = dataClass
        self.x = timestep
 
   
    def fitpolynomial(self, ep, multfac):
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

        
    def temp_term(self, Theta):
        """Calculates the temperature exponent factor for carbon pool respirations
        given a value for Theta parameter.
        """
        temp_term = np.exp(Theta*self.dC.t_mean[self.x])
        return temp_term
 
               
    def acm(self, cf, clma, ceff):
        """Aggregated canopy model (ACM) function
        ------------------------------------------
        Takes a foliar carbon (cf) value, leaf mass per area (clma) and canopy
        efficiency (ceff) and returns the estimated value for Gross Primary 
        Productivity (gpp) of the forest at that time.
        """
        L = cf / clma
        q = self.dC.a3 - self.dC.a4
        gc = (abs(self.dC.phi_d))**(self.dC.a10) / (0.5*self.dC.t_range[self.x]\
             + self.dC.a6*self.dC.R_tot)
        p = ((ceff*L) / gc)*np.exp(self.dC.a8*self.dC.t_max[self.x])
        ci = 0.5*(self.dC.ca + q - p + np.sqrt((self.dC.ca + q - p)**2 \
             -4*(self.dC.ca*q - p*self.dC.a3)))
        E0 = (self.dC.a7*L**2) / (L**2 + self.dC.a9)
        delta = -23.4*np.cos((360.*(self.dC.D[self.x] + 10) / 365.)*\
                (np.pi/180.))*(np.pi/180.)
        s = 24*np.arccos(( - np.tan(self.dC.lat)*np.tan(delta))) / np.pi
        if s >= 24.:
            s = 24.
        elif s <= 0.:
            s = 0.
        else:
            s = s
        gpp = (E0*self.dC.I[self.x]*gc*(self.dC.ca - ci))*(self.dC.a2*s +\
              self.dC.a5) / (E0*self.dC.I[self.x] + gc*(self.dC.ca - ci))          
        return gpp

        
    def phi_onset(self, d_onset, cronset):
        """Leaf onset function (controls labile to foliar carbon transfer) takes 
        d_onset value, cronset value and returns a value for phi_onset.
        """
        releasecoeff = np.sqrt(2.)*cronset / 2.
        magcoeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
        offset = self.fitpolynomial(1+1e-3, releasecoeff)
        phi_onset = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                    np.sin((self.dC.D[self.x] - d_onset + offset) / \
                    self.dC.radconv)*(self.dC.radconv / releasecoeff))**2)
        return phi_onset
 
       
    def phi_fall(self, d_fall, crfall, clspan):
        """Leaf fall function (controls foliar to litter carbon transfer) takes 
        d_fall value, crfall value, clspan value and returns a value for
        phi_fall.
        """
        releasecoeff = np.sqrt(2.)*crfall / 2.
        magcoeff = (np.log(clspan) - np.log(clspan -1.)) / 2.
        offset = self.fitpolynomial(clspan, releasecoeff)
        phi_fall = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff)*np.exp(-( \
                    np.sin((self.dC.D[self.x] - d_fall + offset) / \
                     self.dC.radconv)*self.dC.radconv / releasecoeff)**2)
        return phi_fall        

        
    def dalecv2(self, p):
        """DALECV2 carbon balance model 
        -------------------------------
        evolves carbon pools to the next time step, taking the 6 carbon pool values
        and 17 parameters at time t and evolving them to time t+1. 
        
        phi_on = phi_onset(d_onset, cronset)
        phi_off = phi_fall(d_fall, crfall, clspan)
        gpp = acm(cf, clma, ceff)
        temp = temp_term(Theta)
        
        clab2 = (1 - phi_on)*clab + (1-f_auto)*(1-f_fol)*f_lab*gpp
        cf2 = (1 - phi_off)*cf + phi_on*clab + (1-f_auto)*f_fol*gpp
        cr2 = (1 - theta_roo)*cr + (1-f_auto)*(1-f_fol)*(1-f_lab)*f_roo*gpp
        cw2 = (1 - theta_woo)*cw + (1-f_auto)*(1-f_fol)*(1-f_lab)*(1-f_roo)*gpp
        cl2 = (1-(theta_lit+theta_min)*temp)*cl + theta_roo*cr + phi_off*cf
        cs2 = (1 - theta_som*temp)*cs + theta_woo*cw + theta_min*temp*cl
        """        
        out = algopy.zeros(23, dtype=p)        
        
        phi_on = self.phi_onset(p[17], p[19])
        phi_off = self.phi_fall(p[20], p[21], p[10])
        gpp = self.acm(p[1], p[22], p[16])
        temp = self.temp_term(p[15])
        
        out[0] = (1 - phi_on)*p[0] + (1-p[7])*(1-p[8])*p[18]*gpp
        out[1] = (1 - phi_off)*p[1] + phi_on*p[0] + (1-p[7])*p[8]*gpp
        out[2] = (1 - p[12])*p[2] + (1-p[7])*(1-p[8])*(1-p[18])*p[9]*gpp
        out[3] = (1 - p[11])*p[3] + (1-p[7])*(1-p[8])*(1-p[18])*(1-p[9])*gpp
        out[4] = (1-(p[13]+p[6])*temp)*p[4] + p[12]*p[2] + phi_off*p[1]
        out[5] = (1 - p[14]*temp)*p[5] + p[11]*p[3] + p[6]*temp*p[4]
        out[6:23] = p[6:23]
        return out

        
    def jac_dalecv2(self, p):
        """Using algopy package calculates the jacobian for dalecv2 given a 
        input vector p.
        """
        p = algopy.UTPM.init_jacobian(p)
        return algopy.UTPM.extract_jacobian(self.dalecv2(p)) 
 
   
    def mod_list(self, pvals, lenrun):
        """Creates an array of evolving model values using dalecv2 function.
        Takes a list of initial param values.
        """
        mod_list = np.concatenate((np.array([pvals]),\
                                   np.ones((lenrun, len(pvals)))*-9999.))       
        
        for t in xrange(lenrun):
            mod_list[(t+1)] = self.dalecv2(mod_list[t])
            self.x += 1
        
        self.x -= lenrun
        return mod_list

        
    def linmod_list(self, pvals, lenrun):
        """Creates an array of linearized models (Mi's) taking a list of initial 
        param values and a run length (lenrun).
        """
        mod_list = np.concatenate((np.array([pvals]),\
                                   np.ones((lenrun, len(pvals)))*-9999.))
        matlist = np.ones((lenrun, 23, 23))*-9999.
        
        for t in xrange(lenrun):
            mod_list[(t+1)] = self.dalecv2(mod_list[t])
            matlist[t] = self.jac_dalecv2(mod_list[t])
            self.x += 1
            
        self.x -= lenrun    
        return mod_list, matlist
        

