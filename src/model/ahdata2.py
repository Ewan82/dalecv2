"""dataClass extracting driving data from Alice Holt mainly deciduous forest
required to run dalecv2 model forwards. As a class function will also extract 
observations used in assimilation.
"""
import numpy as np
import os
import re
import matplotlib.mlab as ml
import collections as col

class dalecData( ): 

    def __init__(self, lenrun, startrun=0, k=1,):
        
        #Extract the data
        self.homepath = os.path.expanduser("~")
        self.data = ml.csv2rec(self.homepath+\
                              "/dalecv2/aliceholtdata/ahdata3.csv",\
                              missing='nan')
        self.lenrun = lenrun
        self.startrun = startrun
        self.fluxdata = self.data[startrun:startrun+lenrun]
        self.timestep = np.arange(startrun, startrun+lenrun)
        
        #'I.C. for carbon pools gCm-2'   range
        self.clab = 100.8               # (10,1e3)
        self.cf = 10.0                  # (10,1e3)
        self.cr = 102.0                 # (10,1e3)
        self.cw = 8100.0                # (3e3,3e4)
        self.cl = 100.0                 # (10,1e3) 
        self.cs = 7200.0                # (1e3, 1e5)
        self.clist = np.array([[self.clab,self.cf,self.cr,self.cw,self.cl,\
                                self.cs]])
        
        #'Parameters for optimization'                     range
        self.p1 = 3e-3 #theta_min, cl to cs decomp    (1e-5 - 1e-2)day^-1
        self.p2 = 0.32 #f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p3 = 0.465 #f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.p4 = 0.32 #f_roo, frac GPP to fine roots     (0.01 - 0.5)
        self.p5 = 1.025 #clspan, leaf lifespan             (1.0001 - 5)
        self.p6 = 2e-4 #theta_woo, wood C turnover     (2.5e-5 - 1e-3)day^-1
        self.p7 = 0.007 #theta_roo, root C turnover rate(1e-4 - 1e-2)day^-1
        self.p8 = 0.025 #theta_lit, litter C turnover   (1e-4 - 1e-2)day^-1
        self.p9 = 2e-4 #theta_som, SOM C turnover    (1e-7 - 1e-3)day^-1 
        self.p10 = 0.069 #Theta, temp dependence exp fact(0.018 - 0.08)
        self.p11 = 30. #ceff, canopy efficiency param     (10 - 100)        
        self.p12 = 140. #d_onset, clab release date       (1 - 365) (60,150)
        self.p13 = 0.06 #f_lab, frac GPP to clab        (0.01 - 0.5)
        self.p14 = 27. #cronset, clab release period      (10 - 100)
        self.p15 = 308. #d_fall, date of leaf fall        (1 - 365) (242,332)
        self.p16 = 35. #crfall, leaf fall period          (10 - 100)
        self.p17 = 52. #clma, leaf mass per area          (10 - 400)gCm^-2
  
        self.paramdict = col.OrderedDict([('clab', self.clab), ('cf', self.cf), 
                       ('cr', self.cr), ('cw', self.cw), ('cl', self.cl),
                       ('cs', self.cs), ('theta_min', self.p1), 
                       ('f_auto', self.p2), ('f_fol', self.p3), 
                       ('f_roo', self.p4), ('clspan', self.p5), 
                       ('theta_woo', self.p6), ('theta_roo', self.p7), 
                       ('theta_lit', self.p8), ('theta_som', self.p9), 
                       ('Theta', self.p10), ('ceff', self.p11), 
                       ('d_onset', self.p12), ('f_lab', self.p13), 
                       ('cronset', self.p14), ('d_fall', self.p15), 
                       ('crfall', self.p16), ('clma', self.p17)])
        self.pvals = np.array(self.paramdict.values())  
        self.pvals2 = np.array([  3.24116542e+02,3.31196308e+02,7.76019413e+01,
                                  1.23324661e+03,3.40582359e+02,1.01628683e+04,
                                  9.41000000e-04,4.70000000e-01,2.80000000e-01,
                                  1.60000000e-01,1.01000000e+00,2.60000000e-04,
                                  2.48000000e-03,8.38000000e-03,2.60000000e-06,
                                  1.93000000e-02,2.00000000e+01,1.49000000e+02,
                                  5.06290000e-01,2.00000000e+01,2.90000000e+01,
                                  6.00000000e+01,5.20000000e+01])

        
        #Constants for ACM model 
        #(currently using parameters from williams spreadsheet values)
        self.a2 = 0.0155 #0.0156935
        self.a3 = 1.526 #4.22273 hashed values from reflex experiment paper
        self.a4 = 324.1 #208.868
        self.a5 = 0.2017 #0.0453194
        self.a6 = 1.315 #0.37836
        self.a7 = 2.595 #7.19298
        self.a8 = 0.037 #0.011136
        self.a9 = 0.2268 #2.1001
        self.a10 = 0.9576 #0.789798
        self.phi_d = -2. #max. soil leaf water potential difference
        self.R_tot = 1. #total plant-soil hydrolic resistance
        self.lat = 0.89133965 #latitutde of forest site in radians
        
        #'Daily temperatures degC'
        self.t_mean = self.fluxdata['t_mean'].tolist()*k
        self.t_max = self.fluxdata['t_max'].tolist()*k
        self.t_min = self.fluxdata['t_min'].tolist()*k
        self.t_range = np.array(self.t_max) - np.array(self.t_min)
        
        #'Driving Data'
        self.I = self.fluxdata['i'].tolist()*k #incident radiation
        self.ca = 390.0 #atmospheric carbon    
        self.D = self.fluxdata['day'].tolist()*k #day of year 
        
        #misc
        self.radconv = 365.25 / np.pi
        
        #'Background variances for carbon pools & B matrix'
        self.sigb_clab = 8.36**2 #(self.clab*0.2)**2 #20%
        self.sigb_cf = 11.6**2 #(self.cf*0.2)**2 #20%
        self.sigb_cw = 20.4**2 #(self.cw*0.2)**2 #20%
        self.sigb_cr = 154.**2 #(self.cr*0.2)**2 #20%
        self.sigb_cl = 8.**2 #(self.cl*0.2)**2 #20%
        self.sigb_cs = 1979.4**2 #(self.cs*0.2)**2 #20% 
        self.B = (0.2*np.array([self.pvals]))**2*np.eye(23)
        #MAKE NEW B, THIS IS WRONG!
        
        #'Observartion variances for carbon pools and NEE' 
        self.sigo_clab = (self.clab*0.1)**2 #10%
        self.sigo_cf = (self.cf*0.1)**2 #10%
        self.sigo_cw = (self.cw*0.1)**2 #10%
        self.sigo_cr = (self.cr*0.3)**2 #30%
        self.sigo_cl = (self.cl*0.3)**2 #30%
        self.sigo_cs = (self.cs*0.3)**2 #30% 
        self.sigo_nee = 0.5 #(gCm-2day-1)**2
        self.sigo_lf = 0.2**2
        self.sigo_lw = 0.2**2
        
        self.errdict = {'clab':self.sigo_clab, 'cf':self.sigo_cf,\
                        'cw':self.sigo_cw,'cl':self.sigo_cl,'cr':self.sigo_cr,\
                        'cs':self.sigo_cs, 'nee':self.sigo_nee,\
                        'lf':self.sigo_lf, 'lw':self.sigo_lw}
        

    def assimilation_obs(self, obs_str):
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs', 'lai', 'clab']
        Obslist = re.findall(r'[^,;\s]+', obs_str)
        Obs_dict = {}
        Obs_err_dict = {}
        for ob in Obslist:
            if ob not in possibleobs:
                raise Exception('Invalid observations entered, please check \
                                 function input')
            else:
                Obs_dict[ob] = self.fluxdata[ob]
                Obs_err_dict[ob+'_err'] = (self.fluxdata[ob]/self.fluxdata[ob])\
                                          *self.errdict[ob]
        
        return Obs_dict, Obs_err_dict