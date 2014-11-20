import matplotlib.mlab as mlab
import numpy as np


def data(filename):
    """Extracts data from specified .csv file.
    """
    fluxdata = mlab.csv2rec(filename, missing = 'N/A')
    fluxdata['nee'][fluxdata['nee']>=100]=float('NaN')
    fluxdata['nee'][fluxdata['nee']<=-100]=float('NaN')
   
    lenyear = len(fluxdata)/48
    
    year = np.ones(lenyear)*fluxdata['date'][0].year
    day = np.arange(1, lenyear+1)
    t_mean = np.ones(lenyear)*-9999
    t_max = np.ones(lenyear)*-9999
    t_min = np.ones(lenyear)*-9999
    I = np.ones(lenyear)*-9999
    nee = np.ones(lenyear)*-9999
    
    for x in xrange(0, lenyear):
        t_mean[x] = np.mean(fluxdata['t'][48*x:48*x+48])
        t_max[x] = np.max(fluxdata['t'][48*x:48*x+48])
        t_min[x] = np.min(fluxdata['t'][48*x:48*x+48])
        I[x] = 30*60*1e-6*np.sum(fluxdata['rg'][48*x:48*x+48])
        fill = 0
        qcflag = 0
        for qc in xrange(48*x, 48*x+48):
            if fluxdata['nee'][qc] == float('NaN'):
                fill = fill + 1
            if fluxdata['qc'][qc] == 2:
                qcflag = qcflag + 1
        if fill > 0:
            nee[x] = float('NaN')
        elif qcflag > 4:
            nee[x] = float('NaN')
        else:
            nee[x] = 12*1e-6*30*60*np.sum(fluxdata['nee'][48*x:48*x+48])
            
    return np.array([year, day, t_mean, t_max, t_min, I, nee])
    

def dat_output(filenames, outputname):

    dat = data(filenames[0])
    for x in xrange(1,len(filenames)):
        dat = np.append(dat, data(filenames[x]), axis = 1)
    
    np.savetxt(outputname, dat.T, delimiter=',', fmt='%.3e', header="year,"
    "day, t_mean, t_max, t_min, I, nee", comments='')
    return dat
